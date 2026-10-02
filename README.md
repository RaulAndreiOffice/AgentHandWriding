# Handwritten Math Transcriber

FastAPI service that takes a photo of a page of handwritten math exercises and returns a LaTeX/KaTeX
transcription produced by a Vision-Language Model behind an OpenAI-compatible API (Ollama, vLLM, ...).

```
app/
├── main.py                 # FastAPI app, error format, /health
├── config.py               # settings from env / .env
├── schemas.py              # response models
├── api/routes.py                # POST /api/transcribe-page, POST /api/segment-preview (HTTP only)
└── services/
    ├── segmenter.py             # page -> line crops (component clustering, OpenCV)
    ├── vlm_service.py           # VLM calls + prompts
    ├── verifier.py              # Phase 2: corrections + green/yellow/grey status per line
    └── pipeline_service.py      # segment -> VLM per line -> verify -> merge
scripts/eval_gold.py             # accuracy vs the hw2tex gold set (raw VLM vs verified)
tools/katex_check.mjs            # KaTeX syntax check used by the verifier (Node, optional)
```

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cd tools; npm install; cd ..    # optional: real KaTeX syntax checks in the verifier (needs Node)
```

## 2. Start a vision model

**Ollama** (OpenAI API on `http://localhost:11434/v1`):
```powershell
ollama pull qwen2.5vl:7b
```

**vLLM** (Linux/WSL + GPU):
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
```
then set `VLM_BASE_URL=http://localhost:8001/v1` and `VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`.

## 3. Configure

```powershell
copy .env.example .env
```
Edit `.env` if your endpoint/model differ from the defaults.

## 4. Run

```powershell
uvicorn app.main:app --reload --port 8080
```

## 5. Test via Swagger UI

Open http://localhost:8080/docs.

**Check the segmentation first (no VLM call):**
1. **POST /api/segment-preview** → **Try it out** → **Choose File** (full-page photo) → **Execute**
2. The response is the page with numbered red/blue boxes, one per region the VLM will see.

**Transcribe:**
1. **POST /api/transcribe-page** → **Try it out**
2. Leave `segment` = `true` (set `false` to send the whole page in one call, the old behaviour)
3. Choose the same photo → **Execute** (~1.5–2 s per line on the 4 GB GPU)

```json
{
  "status": "success",
  "latex": "$SIMULARE\ CONSTANTA...$\n\n$1) 5(4 + \sqrt{12}) - \sqrt{300} = 20 (=>)$\n\n...",
  "model": "Qwen/Qwen3-VL-2B-Instruct-FP8",
  "filename": "page.png",
  "processing_time_ms": 29206,
  "segmentation": {"mode": "lines", "regions_detected": 19, "failed_regions": 0, "page_size": [1450, 2115]},
  "verification": {"enabled": true, "green": 12, "yellow": 7, "grey": 0, "corrections": 11},
  "lines": [{"index": 5, "bbox": [253, 477, 274, 88],
             "latex": "$$A(5)=\\begin{pmatrix} 2 & 5 \\\\ -5 & -2 \\end{pmatrix}$$",
             "raw_latex": "$$A(s)=\\begin{pmatrix} 2 & 5 \\\\ -5 & -2 \\end{pmatrix}$$",
             "status": "yellow",
             "issues": [{"kind": "digit_arg", "message": "A(s) -> A(5): page also has A(5)", "fixed": true}],
             "error": null}, ...]
}
```

- `lines[].status`: **green** parses and nothing was corrected or flagged; **yellow** a correction was
  applied or something is flagged, worth a glance; **grey** does not parse, came back empty/illegible,
  or the VLM failed. `raw_latex` is what the VLM returned; `latex` (and the merged `latex`) is verified.

- `segmentation.mode` is `full_page` when fewer than 2 lines were found (or segmentation failed); the page is then sent whole.
- `status` is `partial` when some lines failed; those lines carry an `error` and are left out of `latex`.
- `bbox` is `[x, y, w, h]` in page pixels, to match a line back to the photo.

Or with curl:
```bash
curl -F "file=@page1.jpg" http://localhost:8080/api/transcribe-page
curl -F "file=@page1.jpg" http://localhost:8080/api/segment-preview -o preview.png
```

Errors return `{"status": "error", "detail": "..."}` with: 415 (not an image type), 400 (empty or undecodable file),
413 (over `MAX_UPLOAD_MB`), 502 (VLM unreachable, or every line failed).

## Segmentation

`app/services/segmenter.py` groups connected components into lines (it does not need an empty
row across the page, which grid paper, shadows and sign tables never give you):

1. ink = much darker than the local paper, so grid squares and bleed-through drop out;
2. grid/ruled lines, page edges, the spine, green grader marks and off-page clutter are removed;
3. blobs whose vertical spans overlap form a line; fraction bars bind numerator and denominator;
4. tall strokes (braces, matrix brackets, table rules) bind the lines they fully contain, so
   systems, matrices and sign tables come out as one block, while ordinary lines stay separate.

Use **POST /api/segment-preview** to see the boxes before spending VLM time. Knobs (in `.env`,
pixel values at the 2000 px working scale; unset = automatic from the page's glyph height):

| Setting | Effect |
|---|---|
| `SEG_GRID_FILTER_STRENGTH` (0–1, default 0.5) | Higher: stricter ink contrast, shorter line-removal kernels, larger specks dropped. Lower it if faint pencil disappears; raise it if grid remnants join lines. |
| `SEG_BOX_MERGE_GAP_PX` | Final merge pass: boxes overlapping horizontally merge into one crop when the vertical gap between them is below this (negative = overlap). Set e.g. `15` if matrix rows or fraction parts come out as separate boxes. Auto merges only boxes >50% inside each other (a positive value also joins ordinary lines that close). |
| `SEG_ROW_MERGE_THRESHOLD_PX` | Min vertical overlap for two blobs to count as one line. Auto: 40% of the shorter blob. |
| `SEG_MARGIN_EXCLUDE_FRAC` (default 0.08) | Width of the left/right strips searched for a page edge / spine. |
| `SEG_SPLIT_COLUMNS` (default true), `SEG_COLUMN_GAP_PX` | Split a block of 2+ rows into a left and a right column at an ink-free vertical gutter (auto: 2.5 glyphs wide, 3.5 glyphs of content on each side), e.g. a triangle beside its calculations. Not split when both sides have the same rows (a matrix). |
| `SEG_CLASSIFY_REGIONS` (default true) | Tag regions `table` (a sign table: a long rule crossing a vertical one) or `diagram` (a figure: mostly line art, few pieces, a long oblique side meeting another stroke). |

**Figures, columns and region kinds.** Figures are found first, on the ink *before* the ruled-line
filter (which would remove a flat triangle's long base) minus the notebook grid: a horizontal piece is
grid when its row carries horizontal runs across at least 40% of the page (vertical likewise), while a
pen-drawn base exists only for its own length. (Stroke thickness cannot tell them apart: on a 600 px
screenshot the pen is as thin as the grid.) The triangle test fits the smallest enclosing triangle, so
a label or short mark at a vertex, or a number written on a side, does not break it. A figure is a closed triangle, however flat or
obtuse: a connected set of thin sparse strokes whose convex hull has three corners with ink along
the middle of *each* side (a strike-through covers one side at most, a cursive word's ink lies inside
its hull), a long oblique side, no strokes crossing in the middle (a table), most ink on straight lines
(an apex label or angle arc may touch it). Other polygons (squares, rectangles) are read as lines. The drawing's ink is taken out of the text, its labels (A, B, C, 13)
go with it, and it becomes one `diagram` region, so it is never split between lines nor merged with
the formulas beside it, even when its sides slant under them and there is no clear gutter. The
lines on its left come before it, the lines on its right after it. Its crop is masked: only the
figure and its labels are visible. Remaining side-by-side blocks are split at a clear vertical gutter
and read left column first, then the right column, each re-clustered into lines. `/api/segment-preview` draws tables orange ("T") and diagrams purple ("D").
In the API, `lines[].kind` is `line`, `table` or `diagram`:

- **diagram**: not transcribed (the 2B model reads a triangle as a matrix). It is described in a few
  words → `$\text{[Figura geometrica: Triunghi ABC dreptunghic in A]}$` (a bare placeholder when the
  description looks like LaTeX or is too long), status green, and the crop comes back in
  `lines[].image` as a `data:` URL. The description can get numbers wrong; the image is the reference.
- **table**: transcribed with the line prompt by default. `VLM_TABLE_PROMPT=true` uses a dedicated
  sign-table prompt (`\begin{array}`, `\nearrow`/`\searrow`); with the 2B model it is worse on the gold
  set (it copies the prompt's example, or loops `c|c|c|...`), so it is off; a looping answer falls back
  to the line prompt.

Both detectors are tuned for precision on the gold set (2 triangles and 5 tables found, no false
tags); a missed table or figure is simply transcribed as lines, as before. A figure larger than about a
quarter of the page height is dropped by the page-edge filter and does not appear in any crop.

Line crops are upscaled (up to 2.5x, toward 250k pixels, inside vLLM's 512x512 `max_pixels`)
before they are sent: the 2B model confuses 4/u and 5/s much more on small crops.

The prompts (`app/services/vlm_service.py`) include Romanian bac conventions, matrix rules
(`pmatrix`/`vmatrix`, never flatten) and digit-vs-letter rules. They must fit vLLM's
`--max-model-len 1024` together with the image and `VLM_MAX_TOKENS`; `tests/test_prompts.py`
guards the length, so re-measure the token count if you extend them.

Known limits: a long diagonal strike-through binds the lines it crosses; a grader's long arrow can
join two lines; two side-by-side columns are read as one wide line per row; the curved edge of a
photographed page can widen a nearby line's box.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
```

- `tests/test_api.py`: the endpoints with a scripted fake VLM (no vLLM needed): success, `partial`,
  all-lines-failed → 502, full-page fallback, `segment=false`, 400/413/415, `/api/segment-preview`.
- `tests/test_segmenter.py`: segmentation on synthetic pages (plain and grid paper, a fraction,
  EXIF rotation, fallbacks, params) and the merge step.
- The same file also checks every page in the hw2tex gold set (at least 8 regions, none taller than
  25% of the page). It is skipped when the folder is missing; point `HW2TEX_GOLD_DIR` at it.
- `tests/test_verifier.py`: every correction rule, what must *not* be corrected, and the statuses
  (KaTeX tests are skipped without Node).
- `tests/test_prompts.py`, `tests/test_eval_gold.py`: prompt rules/length, metric normalisation.

## Verification (Phase 2)

`app/services/verifier.py` runs after the VLM on all lines of a page (`VERIFY_TRANSCRIPTIONS=false`
turns it off). It is deterministic: no extra model calls.

| Rule | Example | Effect on status |
|---|---|---|
| Missing / unclosed delimiters; labels and Romanian words stay text | `a) f'(x) ≥ 0 pt. x ∈ [0,1]` → `a) $f'(x) \geq 0$ pt. $x \in [0,1]$` | formatting, stays green |
| Unicode math, `=>`, `\left( array \right)`, `.` as product | `x²` → `x^{2}`, → `pmatrix` | formatting, stays green |
| Function argument read as a letter, with page evidence | `A(s)` → `A(5)` when the page also has `A(5)`; flagged only without evidence | yellow |
| Lone `l` / `\ell` / `t` for the digit 1 (not `\ln`, `\lim`, `t_1`, `2t`, `f(t)`; never `t` when the page defines t) | `x \in [-l, l]` → `x \in [-1, 1]` | yellow |
| Misread arrows, composition law | `(=)`, `c=)` → `\Leftrightarrow`; `x o y` → `x \circ y` | yellow |
| Grade mark | `21 \quad A^r` → `21 \quad \text{,,A''}` | yellow |
| One-row / ragged matrix, `\text{[illegible]}` | flagged | yellow |
| KaTeX parse error (Node + `tools/`), else unbalanced braces / environments / `\left`-`\right` | flagged | grey |

## Targeted re-ask (agentic second pass)

`app/services/reask.py` runs after the verifier on **yellow and grey lines only** (a page whose lines
are all green makes no extra VLM call), with at most `VLM_CONCURRENCY` calls in flight:

1. **Re-transcription** of grey lines (KaTeX error, empty, illegible, failed call) and malformed
   matrices, with a strict output schema and a description of what went wrong. Kept only if the
   verifier rates the new text better.
2. **Closed questions** for each open ambiguity on a yellow line (at most
   `REASK_MAX_QUESTIONS_PER_LINE`): `s_vs_5:A(s)` → "is the character inside the parentheses the digit
   5 or the letter s? Answer 5 or s", `arrow:(=)` (A: ⇔, B: ⇒, C: =), `o_vs_circ`, `grade_mark`.
   Only the **text** of the answer is used: no logprobs (under WSL they go through a
   `torch.compile`d sampler helper that needs `nvcc`, and the first such request kills vLLM) and no
   choice constraint. An answer that is one of the options is used when it **decides a flag** the
   verifier left open or **confirms** the verifier's correction (the line turns green). It **never
   reverses** a correction: asked about its own misreadings, the 2B model mostly repeats them (arrows:
   29 of 44 answers contradicted the reading the gold confirms). Lone `l`/`t` vs `1` is not asked at all
   (0/3 right on tema17).

Each line reports its re-asks in `lines[].reasks`; `verification.reask_calls` / `reask_accepted`
summarise them. `REASK_ENABLED=false` turns the pass off.

## Evaluation against the gold set

```powershell
python scripts/eval_gold.py                                   # all gold crops with verified regions (needs vLLM)
python scripts/eval_gold.py --ids corectare_tema17_p06_ex2_matrici
python scripts/eval_gold.py --cached-only --report scripts/out/report.json   # re-score, no vLLM
python scripts/eval_gold.py --reask --report scripts/out/report_reask.json    # + re-ask pass (vLLM, cached)
```

With `--reask` it adds a third column (verified + re-ask), the error by status after re-ask, and per
ambiguity family how many questions were asked / accepted. Re-ask answers are cached in
`scripts/.cache/reask/`.

Last run (42 crops, 166 verified regions): CER raw 42.7% → verified 42.3% → re-ask 41.1%; the re-ask
gain is one re-transcribed region (net −13 characters without it). 67 extra calls, 11 answers used.

Scores only regions marked `"verified": true` in the hw2tex `manifest.json` (the rest is output of
an older pipeline). Our lines are matched to gold regions by bounding box; both sides are normalised
the same way (delimiters, spacing, `\geq`/`\ge`/`≥`, `pmatrix` vs `\left(\begin{matrix}`, `(=)` vs
`\Leftrightarrow`, which the gold writes both ways) so only content counts. It prints CER (character
error rate), TER (LaTeX-token error rate) and exact matches for raw VLM vs verified output, the error by
line status, and the regions the verifier changed. Raw VLM output is cached in `scripts/.cache/`.

## Extending

The route only depends on `get_pipeline()`; further steps go in
`TranscriptionPipeline.transcribe()` (`app/services/pipeline_service.py`). Each line keeps its bbox,
so an agent can re-crop a yellow/grey line and ask the VLM again.
