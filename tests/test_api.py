"""HTTP contract of /api/transcribe-page and /api/segment-preview, with a fake VLM."""

import io

from PIL import Image

from app.config import get_settings
from app.main import app
from app.services.pipeline_service import TranscriptionPipeline, get_pipeline
from tests.conftest import make_settings, png_bytes


def post_page(client, data: bytes, name: str = "page.png", mime: str = "image/png", **params):
    return client.post("/api/transcribe-page", params=params, files={"file": (name, data, mime)})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_lines_success(client, fake_vlm, lined_page):
    r = post_page(client, lined_page)
    assert r.status_code == 200
    body = r.json()
    seg = body["segmentation"]
    assert body["status"] == "success"
    assert body["model"] == "fake-vlm"
    assert seg["mode"] == "lines"
    assert seg["regions_detected"] == len(body["lines"]) == fake_vlm.line_calls == 5
    assert seg["failed_regions"] == 0
    assert fake_vlm.page_calls == 0
    # fences stripped, regions joined top to bottom with blank lines
    assert body["latex"] == "\n\n".join(f"$line{i}$" for i in range(5))
    assert [line["index"] for line in body["lines"]] == list(range(5))
    tops = [line["bbox"][1] for line in body["lines"]]
    assert tops == sorted(tops)


def test_verification_fields(client, fake_vlm, lined_page):
    fake_vlm.overrides = {1: r"$A(s) = 2$", 2: r"$A(5) = \frac{1}{2$"}
    body = post_page(client, lined_page).json()
    v = body["verification"]
    assert v["enabled"] is True
    assert (v["green"], v["yellow"], v["grey"], v["corrections"]) == (3, 1, 1, 1)
    line1, line2 = body["lines"][1], body["lines"][2]
    assert line1["raw_latex"] == r"$A(s) = 2$" and line1["latex"] == r"$A(5) = 2$"
    assert line1["status"] == "yellow"
    assert line1["issues"][0]["kind"] == "digit_arg" and line1["issues"][0]["fixed"] is True
    assert line2["status"] == "grey" and line2["issues"][0]["kind"] == "syntax"
    assert "$A(5) = 2$" in body["latex"]  # the merged document uses the verified text
    assert body["lines"][0]["raw_latex"].startswith("```")  # raw keeps the fences the pipeline strips


def test_verification_can_be_disabled(client, fake_vlm, lined_page):
    settings = make_settings(verify_transcriptions=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: TranscriptionPipeline(fake_vlm, settings)
    body = post_page(client, lined_page).json()
    assert body["verification"] == {"enabled": False, "green": 0, "yellow": 0, "grey": 0, "corrections": 0,
                                    "reask_calls": 0, "reask_accepted": 0}
    assert all(line["status"] is None and line["issues"] == [] for line in body["lines"])


def test_partial_when_some_lines_fail(client, fake_vlm, lined_page):
    fake_vlm.fail = {0, 3}
    body = post_page(client, lined_page).json()
    assert body["status"] == "partial"
    assert body["segmentation"]["failed_regions"] == 2
    errors = [line["error"] for line in body["lines"]]
    assert errors[0] == "boom 0" and errors[3] == "boom 3"
    assert errors[1] is None
    assert "$line0$" not in body["latex"] and "$line1$" in body["latex"]


def test_all_lines_fail_is_502(client, fake_vlm, lined_page):
    fake_vlm.fail = set(range(100))
    r = post_page(client, lined_page)
    assert r.status_code == 502
    assert r.json()["status"] == "error"
    assert r.json()["detail"].startswith("All 5 line transcriptions failed")


def test_blank_page_falls_back_to_full_page(client, fake_vlm):
    blank = png_bytes(Image.new("RGB", (800, 600), "white"))
    body = post_page(client, blank).json()
    assert body["segmentation"]["mode"] == "full_page"
    assert body["segmentation"]["regions_detected"] == 1
    assert body["latex"] == "PAGE"
    assert fake_vlm.page_calls == 1 and fake_vlm.line_calls == 0


def test_segment_false_sends_whole_page(client, fake_vlm, lined_page):
    body = post_page(client, lined_page, segment="false").json()
    assert body["segmentation"] == {"mode": "full_page", "regions_detected": 1,
                                    "failed_regions": 0, "page_size": [0, 0]}
    assert body["latex"] == "PAGE"
    assert fake_vlm.page_calls == 1 and fake_vlm.line_calls == 0


def test_full_page_vlm_error_is_502(client, fake_vlm, lined_page):
    fake_vlm.fail_page = True
    r = post_page(client, lined_page, segment="false")
    assert r.status_code == 502
    assert r.json() == {"status": "error", "detail": "page boom"}


def test_unsupported_type_is_415(client):
    r = post_page(client, b"hello", name="a.txt", mime="text/plain")
    assert r.status_code == 415
    assert r.json()["status"] == "error"


def test_empty_file_is_400(client):
    r = post_page(client, b"")
    assert r.status_code == 400
    assert r.json()["detail"] == "Uploaded file is empty."


def test_undecodable_image_is_400(client):
    r = post_page(client, b"not really a png")
    assert r.status_code == 400
    assert r.json()["detail"] == "File could not be decoded as an image."


def test_oversized_file_is_413(client, lined_page):
    app.dependency_overrides[get_settings] = lambda: make_settings(max_upload_mb=0)
    r = post_page(client, lined_page)
    assert r.status_code == 413


def test_segment_preview_returns_png(client, fake_vlm, lined_page):
    r = client.post("/api/segment-preview", files={"file": ("page.png", lined_page, "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == Image.open(io.BytesIO(lined_page)).size
    assert fake_vlm.line_calls == fake_vlm.page_calls == 0  # no VLM time spent
