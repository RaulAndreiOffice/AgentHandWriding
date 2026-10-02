"""Score raw VLM output vs verified (Phase 2) output against the hw2tex gold set.

    python scripts/eval_gold.py                      # every gold crop with verified regions
    python scripts/eval_gold.py --ids corectare_tema17_p06_ex2_matrici
    python scripts/eval_gold.py --limit 10 --report scripts/out/report.json

Only regions marked "verified": true in manifest.json are scored (the others are
output of an older pipeline, not ground truth). Our lines are matched to a gold
region by bounding box (line centre inside the gold box, else largest overlap),
then both texts are normalised the same way (delimiters, spacing, synonyms such
as \\geq/\\ge/≥, pmatrix vs \\left(\\begin{matrix}) so that only content counts.

Metrics, micro-averaged over regions:
  CER  character error rate  = edit distance / gold length (normalised strings)
  TER  token error rate      = same over LaTeX tokens (\\cmd, digit, letter, symbol)
  exact                      = share of regions whose normalised text matches exactly

Raw VLM lines are cached in scripts/.cache/<crop id>.json, so the verifier can
be re-scored without vLLM; pass --refresh to transcribe again (needs vLLM).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.verifier import verify_page  # noqa: E402

DEFAULT_GOLD = Path(r"D:\Worck\HandWriding_ai\hw2tex\eval\gold")
STATUS_ORDER = {"green": 0, "yellow": 1, "grey": 2, None: 3}

# ------------------------------------------------------------- normalisation

_DROP = re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big|displaystyle|quad|qquad|nonumber)(?![A-Za-z])"
                   r"|\\[,;:! ]|\$")
_SYNONYMS = [
    (r"\\geq(?![A-Za-z])|≥", r"\\ge"), (r"\\leq(?![A-Za-z])|≤", r"\\le"), (r"\\neq(?![A-Za-z])|≠", r"\\ne"),
    # The gold writes the student's "<=>" both as \Leftrightarrow and literally as "(=)" (before "=)").
    (r"\\Leftrightarrow(?![A-Za-z])|\\iff(?![A-Za-z])|⇔|<=>|\(\s*=\s*\)", "⇔"),
    # ...and the student's "=>" both as \Rightarrow and literally as "=)".
    (r"\\Rightarrow(?![A-Za-z])|\\implies(?![A-Za-z])|⇒|=>|=\)(?=\s)", "⇒"),
    # The grade mark: \text{,,A''}, ,,A'', "A", "A ", „A”
    (r"\\text\s*\{\s*,,A''\s*\}|,,A''|\"A\s*\"|„A”|“A”", "⟨A⟩"),
    (r"\\text\s*\{([^{}]*)\}", r"\1"),
    (r"∈", r"\\in"), (r"∞", r"\\infty"), (r"·", r"\\cdot"), (r"×", r"\\times"), (r"→", r"\\to"),
    (r"\\rightarrow(?![A-Za-z])", r"\\to"), (r"²", "^2"), (r"³", "^3"), (r"ℝ", r"\\mathbb{R}"),
    (r"\\[dt]frac(?![A-Za-z])", r"\\frac"), (r"\\vert(?![A-Za-z])", "|"),
    (r"\\operatorname\{([^{}]*)\}", r"\1"), (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\begin\{array\}\{[^{}]*\}", r"\\begin{matrix}"), (r"\\end\{array\}", r"\\end{matrix}"),
]
_ENVS = [(r"\(\s*\\begin\{matrix\}", r"\\begin{pmatrix}"), (r"\\end\{matrix\}\s*\)", r"\\end{pmatrix}"),
         (r"\|\s*\\begin\{matrix\}", r"\\begin{vmatrix}"), (r"\\end\{matrix\}\s*\|", r"\\end{vmatrix}")]
_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")


_FIGURE = re.compile(r"\$?\\text\s*\{\s*\[Figura geometrica[^\]]*\]\s*\}\$?")


def normalise(tex: str) -> str:
    # A figure placeholder is metadata, not a transcription: the gold never transcribes figures.
    s = _FIGURE.sub(" ", tex)
    # Protect the row separator first: "\\ -3" must not lose a backslash to the "\ " spacing rule.
    s = s.replace("\\\\", " ⏎ ")
    for pat, rep in _SYNONYMS:
        s = re.sub(pat, rep, s)
    s = _DROP.sub(" ", s)
    for pat, rep in _ENVS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"([\^_])\{(\S)\}", r"\1\2", s)  # x^{2} == x^2
    return re.sub(r"\s+", "", s)


def tokens(norm: str) -> list[str]:
    return _TOKEN.findall(norm)


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ matching


def _center_inside(line_bbox, g) -> bool:
    x, y, w, h = line_bbox
    cx, cy = x + w / 2, y + h / 2
    return g["x"] <= cx <= g["x"] + g["w"] and g["y"] <= cy <= g["y"] + g["h"]


def _overlap_share(line_bbox, g) -> float:
    x, y, w, h = line_bbox
    ix = max(0, min(x + w, g["x"] + g["w"]) - max(x, g["x"]))
    iy = max(0, min(y + h, g["y"] + g["h"]) - max(y, g["y"]))
    return ix * iy / max(1, w * h)


def match_lines(lines: list[dict], regions: list[dict]) -> dict[int, list[int]]:
    """gold region index -> indices of our lines assigned to it (each line at most once)."""
    out: dict[int, list[int]] = {k: [] for k in range(len(regions))}
    for li, line in enumerate(lines):
        inside = [k for k, r in enumerate(regions) if _center_inside(line["bbox"], r["bbox"])]
        if inside:
            out[inside[0]].append(li)
            continue
        best = max(range(len(regions)), key=lambda k: _overlap_share(line["bbox"], regions[k]["bbox"]), default=None)
        if best is not None and _overlap_share(line["bbox"], regions[best]["bbox"]) >= 0.3:
            out[best].append(li)
    return out


# ---------------------------------------------------------------- transcribe


async def transcribe_raw(image: Path) -> list[dict]:
    """Segment + VLM, verification off (the verifier is applied by this script)."""
    from app.config import get_settings
    from app.services.pipeline_service import TranscriptionPipeline
    from app.services.vlm_service import VLMService

    settings = get_settings().model_copy(update={"verify_transcriptions": False})
    vlm = VLMService(settings)
    try:
        result = await TranscriptionPipeline(vlm, settings).transcribe(image.read_bytes(), "image/png")
    finally:
        await vlm.client.close()  # inside the loop; asyncio.run() closes it per crop
    return [{"index": l.index, "bbox": l.bbox, "latex": l.raw_latex, "error": l.error, "kind": l.kind}
            for l in result.lines]


def kinds_of(lines: list[dict]) -> list[str]:
    """Region kinds for the verifier (tables get the monotony-arrow rule)."""
    return [l.get("kind", "line") for l in lines]


class CachingVLM:
    """The re-ask calls of one crop, cached on disk (key: crop image hash + question)."""

    def __init__(self, path: Path, refresh: bool, settings):
        self.path, self.settings = path, settings
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not refresh else {}
        self._vlm = None
        self.misses = 0

    def _real(self):
        if self._vlm is None:
            from app.services.vlm_service import VLMService
            self._vlm = VLMService(self.settings)
        return self._vlm

    @staticmethod
    def _key(kind: str, image_bytes: bytes, text: str) -> str:
        return f"{kind}:{hashlib.sha1(image_bytes).hexdigest()[:12]}:{text}"

    async def classify(self, image_bytes, mime_type, question, options):
        key = self._key("classify", image_bytes, question)
        if key not in self.data:
            self.misses += 1
            self.data[key] = await self._real().classify(image_bytes, mime_type, question, options)
        cached = self.data[key]
        return cached[0] if isinstance(cached, list) else cached  # older caches stored [answer, confidence]

    async def retranscribe_line(self, image_bytes, mime_type, problem, previous=None):
        key = self._key("retranscribe", image_bytes, f"{problem}|{previous}")
        if key not in self.data:
            self.misses += 1
            self.data[key] = await self._real().retranscribe_line(image_bytes, mime_type, problem, previous)
        return self.data[key]

    async def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        if self._vlm is not None:
            await self._vlm.client.close()


async def reask_crop(crop_id: str, image: Path, lines: list[dict], cache: Path, refresh: bool):
    """Run the pipeline's re-ask pass on cached lines: (LineTranscriptions, events), or None
    when today's segmentation no longer matches the cached lines (re-run with --refresh).
    Single-region pages come back verified but not re-asked, like in the pipeline."""
    from app.config import get_settings
    from app.services.pipeline_service import LineTranscription
    from app.services.reask import Reasker
    from app.services.segmenter import segment_page

    settings = get_settings()
    seg = segment_page(image.read_bytes(), settings.segmentation_params())
    out = [LineTranscription(l["index"], l["bbox"], latex=l["latex"], error=l["error"], kind=l.get("kind", "line"),
                             raw_latex=l["latex"], source_latex=l["latex"]) for l in lines]
    kinds = kinds_of(lines)

    async def verify(texts, errors, decisions):
        return verify_page(texts, errors, None, decisions, kinds)

    if seg.mode != "lines":
        # Single-region page: the pipeline does not re-ask these either; score the verified text.
        for line, v in zip(out, await verify([l.latex for l in out], [l.error for l in out], None)):
            line.latex, line.status, line.issues = v.latex, v.status, v.issues
        return out, []
    if [r.bbox.as_list() for r in seg.regions] != [l["bbox"] for l in lines]:
        return None
    # Like the pipeline: only ordinary lines are re-asked (its prompts are line prompts).
    crops = {r.index: (r.image_bytes, r.mime_type) for r in seg.regions if r.kind == "line"}

    vlm = CachingVLM(cache / "reask" / f"{crop_id}.json", refresh, settings)
    try:
        events = await Reasker(vlm, settings).run(out, crops, verify)
    finally:
        await vlm.close()
    return out, events


def current_regions(image: Path) -> list[tuple[list[int], str]]:
    """Today's segmentation of the crop, (bbox, kind) per region (to tell whether a cached
    transcription still applies, and to know which cached lines are tables)."""
    from app.config import get_settings
    from app.services.segmenter import segment_page

    seg = segment_page(image.read_bytes(), get_settings().segmentation_params())
    return [(r.bbox.as_list(), r.kind) for r in seg.regions] if seg.mode == "lines" else []


def load_lines(crop_id: str, image: Path, cache: Path, refresh: bool, offline: bool = False) -> list[dict] | None:
    """Raw VLM lines for a crop: from the cache when it matches today's segmentation,
    else transcribed again (None when offline and the cache is stale)."""
    path = cache / f"{crop_id}.json"
    if path.exists() and not refresh:
        lines = json.loads(path.read_text(encoding="utf-8"))
        regions = current_regions(image)
        if not regions or [b for b, _ in regions] == [l["bbox"] for l in lines]:  # single-region: not re-segmented
            for line, (_, kind) in zip(lines, regions):
                line.setdefault("kind", kind)  # caches written before kinds were recorded
            return lines
        if offline:
            return None
        print(f"  ~ {crop_id}: segmentation changed, transcribing again")
    lines = asyncio.run(transcribe_raw(image))
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")
    return lines


# ------------------------------------------------------------------- scoring


@dataclass
class Totals:
    char_err: int = 0
    char_len: int = 0
    tok_err: int = 0
    tok_len: int = 0
    exact: int = 0
    regions: int = 0

    def add(self, pred: str, gold: str) -> tuple[int, int]:
        p, g = normalise(pred), normalise(gold)
        ce, te = edit_distance(p, g), edit_distance(tokens(p), tokens(g))
        self.char_err += ce
        self.char_len += len(g)
        self.tok_err += te
        self.tok_len += len(tokens(g))
        self.exact += p == g
        self.regions += 1
        return ce, len(g)

    @property
    def cer(self) -> float:
        return self.char_err / max(1, self.char_len)

    @property
    def ter(self) -> float:
        return self.tok_err / max(1, self.tok_len)

    def row(self) -> str:
        return f"CER {self.cer:6.1%}  TER {self.ter:6.1%}  exact {self.exact:3d}/{self.regions:<3d}"


@dataclass
class Report:
    raw: Totals = field(default_factory=Totals)
    verified: Totals = field(default_factory=Totals)
    reask: Totals = field(default_factory=Totals)
    by_status: dict[str, Totals] = field(default_factory=dict)
    by_status_reask: dict[str, Totals] = field(default_factory=dict)
    crops: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    reask_examples: list[dict] = field(default_factory=list)
    reask_events: list[dict] = field(default_factory=list)


def score_crop(crop: dict, lines: list[dict], report: Report, reasked=None) -> dict:
    """reasked: the LineTranscriptions after the re-ask pass (None = not run)."""
    regions = [r for r in crop["regions"] if r.get("verified") and r["latex"].strip()]
    verified = verify_page([l["latex"] for l in lines], [l["error"] for l in lines], kinds=kinds_of(lines))
    matches = match_lines(lines, regions)
    raw_t, ver_t, rsk_t = Totals(), Totals(), Totals()
    for k, region in enumerate(regions):
        idx, gold = matches[k], region["latex"]
        raw = "\n".join(lines[i]["latex"] for i in idx if not lines[i]["error"])
        ver = "\n".join(verified[i].latex for i in idx)
        for totals in (raw_t, report.raw):
            totals.add(raw, gold)
        for totals in (ver_t, report.verified):
            totals.add(ver, gold)
        status = max((verified[i].status for i in idx), key=STATUS_ORDER.get, default=None) or "unmatched"
        report.by_status.setdefault(status, Totals()).add(ver, gold)
        if normalise(raw) != normalise(ver):
            report.examples.append({"crop": crop["id"], "gold": gold, "raw": raw, "verified": ver,
                                    "raw_err": edit_distance(normalise(raw), normalise(gold)),
                                    "verified_err": edit_distance(normalise(ver), normalise(gold))})
        if reasked is not None:
            rsk = "\n".join(reasked[i].latex for i in idx)
            for totals in (rsk_t, report.reask):
                totals.add(rsk, gold)
            status = max((reasked[i].status for i in idx), key=STATUS_ORDER.get, default=None) or "unmatched"
            report.by_status_reask.setdefault(status, Totals()).add(rsk, gold)
            if normalise(rsk) != normalise(ver):
                report.reask_examples.append({"crop": crop["id"], "gold": gold, "verified": ver, "reask": rsk,
                                              "verified_err": edit_distance(normalise(ver), normalise(gold)),
                                              "reask_err": edit_distance(normalise(rsk), normalise(gold))})
    final = reasked if reasked is not None else verified
    statuses = [v.status for v in final]
    summary = {"id": crop["id"], "regions": len(regions), "lines": len(lines),
               "raw_cer": raw_t.cer, "verified_cer": ver_t.cer, "reask_cer": rsk_t.cer if reasked else None,
               "green": statuses.count("green"), "yellow": statuses.count("yellow"), "grey": statuses.count("grey"),
               "corrections": sum(v.corrections for v in verified),
               "reask_calls": sum(len(l.reasks) for l in reasked) if reasked else 0}
    report.crops.append(summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--ids", nargs="*", help="crop ids to score (default: all with verified regions)")
    ap.add_argument("--limit", type=int, help="score at most this many crops")
    ap.add_argument("--cache", type=Path, default=ROOT / "scripts" / ".cache")
    ap.add_argument("--refresh", action="store_true", help="re-run the VLM even when cached")
    ap.add_argument("--cached-only", action="store_true", help="skip crops that are not cached (no vLLM needed)")
    ap.add_argument("--report", type=Path, help="write a JSON report here")
    ap.add_argument("--examples", type=int, default=8, help="print this many changed regions")
    ap.add_argument("--reask", action="store_true",
                    help="also run the targeted re-ask pass on yellow/grey lines (needs vLLM unless cached)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    manifest = json.loads((args.gold_dir / "manifest.json").read_text(encoding="utf-8"))
    crops = [c for c in manifest["crops"] if any(r.get("verified") and r["latex"].strip() for r in c["regions"])]
    if args.ids:
        crops = [c for c in crops if c["id"] in set(args.ids)]
    if args.cached_only:
        crops = [c for c in crops if (args.cache / f"{c['id']}.json").exists()]
    crops = crops[: args.limit] if args.limit else crops
    if not crops:
        print("No gold crops selected.")
        return 1

    report = Report()
    print(f"{'crop':42s} {'reg':>3s} {'raw CER':>8s} {'ver CER':>8s} {'reask':>7s}  green/yellow/grey  fixes asks")
    for n, crop in enumerate(crops, 1):
        image = args.gold_dir / crop["image"]
        lines = load_lines(crop["id"], image, args.cache, args.refresh, offline=args.cached_only)
        if lines is None:
            print(f"  ! {crop['id']}: cached transcription is from an older segmentation; skipped (--cached-only)")
            continue
        reasked = None
        if args.reask:
            result = asyncio.run(reask_crop(crop["id"], image, lines, args.cache, args.refresh))
            if result is None:
                # Keep the comparison on the same regions: score the verified text in the re-ask column.
                print(f"  ! {crop['id']}: segmentation changed since the cache was made; re-ask skipped, "
                      f"verified text counted (run with --refresh)")
                verified = verify_page([l["latex"] for l in lines], [l["error"] for l in lines], kinds=kinds_of(lines))
                reasked = [SimpleNamespace(latex=v.latex, status=v.status, reasks=[]) for v in verified]
            else:
                reasked, events = result
                report.reask_events.extend(vars(e) | {"crop": crop["id"]} for e in events)
        s = score_crop(crop, lines, report, reasked)
        rsk = f"{s['reask_cer']:7.1%}" if s["reask_cer"] is not None else "      -"
        print(f"{s['id'][:42]:42s} {s['regions']:3d} {s['raw_cer']:8.1%} {s['verified_cer']:8.1%} {rsk}  "
              f"{s['green']:5d}/{s['yellow']:3d}/{s['grey']:3d}  {s['corrections']:5d} {s['reask_calls']:4d}"
              f"   [{n}/{len(crops)}]", flush=True)

    print("\nOverall (verified gold regions only, micro-averaged)")
    print(f"  raw VLM   : {report.raw.row()}")
    print(f"  verified  : {report.verified.row()}")
    if report.reask.regions:
        print(f"  + re-ask  : {report.reask.row()}")
    delta = report.raw.cer - report.verified.cer
    print(f"  CER change raw -> verified    : {delta:+.1%} points")
    if report.reask.regions:
        delta = report.verified.cer - report.reask.cer
        print(f"  CER change verified -> re-ask: {delta:+.1%} points")
    print("\nVerified output by line status (worst status among a region's lines)")
    for status in ("green", "yellow", "grey", "unmatched"):
        if status in report.by_status:
            print(f"  {status:9s}: {report.by_status[status].row()}")
    if report.reask.regions:
        print("After re-ask")
        for status in ("green", "yellow", "grey", "unmatched"):
            if status in report.by_status_reask:
                print(f"  {status:9s}: {report.by_status_reask[status].row()}")
        ev = report.reask_events
        print(f"\nRe-ask: {len(ev)} call(s), {sum(e['accepted'] for e in ev)} accepted")
        families: dict[str, list[dict]] = {}
        for e in ev:
            fam = "retranscribe" if e["kind"] == "retranscribe" else re.sub(r":.*$", "", e["key"] or "?")
            families.setdefault(fam, []).append(e)
        for fam, items in sorted(families.items()):
            print(f"  {fam:16s}: {len(items):3d} asked, {sum(e['accepted'] for e in items):3d} accepted")
        moved = sorted(report.reask_examples, key=lambda e: e["verified_err"] - e["reask_err"], reverse=True)
        for title, items in (("improved", [e for e in moved if e["reask_err"] < e["verified_err"]]),
                             ("made worse", [e for e in moved if e["reask_err"] > e["verified_err"]][::-1])):
            if items:
                print(f"\nRegions the re-ask {title}: {len(items)}")
                for e in items[: args.examples]:
                    print(f"  [{e['crop']}] edits {e['verified_err']} -> {e['reask_err']}")
                    print(f"    gold    : {e['gold'][:160]}")
                    print(f"    verified: {e['verified'][:160]}")
                    print(f"    re-ask  : {e['reask'][:160]}")

    changed = sorted(report.examples, key=lambda e: e["raw_err"] - e["verified_err"], reverse=True)
    if args.examples and changed:
        print(f"\nRegions the verifier changed (best {min(args.examples, len(changed))} by error reduction)")
        for e in changed[: args.examples]:
            print(f"  [{e['crop']}] edits {e['raw_err']} -> {e['verified_err']}")
            print(f"    gold    : {e['gold'][:160]}")
            print(f"    raw     : {e['raw'][:160]}")
            print(f"    verified: {e['verified'][:160]}")
        worse = [e for e in changed if e["verified_err"] > e["raw_err"]]
        if worse:
            print(f"\n{len(worse)} region(s) got worse; first: [{worse[0]['crop']}] "
                  f"{worse[0]['raw'][:100]!r} -> {worse[0]['verified'][:100]!r}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "raw": vars(report.raw) | {"cer": report.raw.cer, "ter": report.raw.ter},
            "verified": vars(report.verified) | {"cer": report.verified.cer, "ter": report.verified.ter},
            "reask": vars(report.reask) | {"cer": report.reask.cer, "ter": report.reask.ter},
            "by_status": {k: vars(v) | {"cer": v.cer} for k, v in report.by_status.items()},
            "by_status_after_reask": {k: vars(v) | {"cer": v.cer} for k, v in report.by_status_reask.items()},
            "crops": report.crops, "changed_regions": changed,
            "reask_changed_regions": report.reask_examples, "reask_events": report.reask_events,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
