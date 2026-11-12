"""UI-agnostic bridge between the Streamlit client and the owagent pipeline.

Keeps all pipeline logic out of the UI layer so it can also be called/tested
from plain Python. No streamlit import here on purpose.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path
from types import SimpleNamespace

from .cli import OUTPUTS, ROOT, SAMPLES, process_video
from .config import load_config, pixel_roi, validate
from .selftest import generate
from .util import probe, run_ffmpeg


def load_base_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.json"
    return load_config(str(cfg_path) if cfg_path.is_file() else None)


class _LogCapture(io.StringIO):
    """Forward every printed log line to a UI callback while still buffering it."""

    def __init__(self, on_line, original_stdout):
        super().__init__()
        self._on_line = on_line
        self._original = original_stdout
        self._busy = False

    def write(self, s: str) -> int:
        super().write(s)
        text = s.rstrip()
        if text and not self._busy:
            self._busy = True
            try:
                # Restore real stdout while the callback runs, otherwise a print()
                # inside the UI callback would re-enter this writer forever.
                with contextlib.redirect_stdout(self._original):
                    self._on_line(text)
            finally:
                self._busy = False
        return len(s)


def run_pipeline(video: str | Path, cfg: dict, *, dry_run: bool = False,
                 work_dir: str | Path | None = None,
                 out_path: str | Path | None = None, on_log=None) -> dict:
    """Run the full detect -> score -> render -> report pipeline."""
    validate(cfg)
    capture = _LogCapture(on_log or (lambda _line: None), sys.stdout)
    with contextlib.redirect_stdout(capture):
        result = process_video(video, SimpleNamespace(dry_run=bool(dry_run)), cfg,
                               work_dir=work_dir, out_path=out_path)
    return result


def build_config(base: dict, *, roi, threshold, fps, pre, post, gap, top_k,
                 no_audio, vertical, burn, bgm, llm) -> dict:
    """Translate UI widget values into a validated pipeline config."""
    c = copy.deepcopy(base)
    x, y, w, h = [float(v) for v in roi]
    x = min(max(x, 0.0), 0.99)
    y = min(max(y, 0.0), 0.99)
    w = min(max(w, 0.01), 1.0 - x)
    h = min(max(h, 0.01), 1.0 - y)  # clamp instead of crashing the page
    c["roi"] = {"x": x, "y": y, "w": w, "h": h}
    c["detect"]["threshold"] = float(threshold)
    c["sample"]["fps"] = int(fps)
    c["clip"].update({"pre_roll": float(pre), "post_roll": float(post),
                      "merge_gap": float(gap)})
    c["select"]["top_k"] = int(top_k)
    c["audio"]["enabled"] = not bool(no_audio)
    c["render"]["vertical"] = bool(vertical)
    c["render"]["burn_subtitles"] = bool(burn)
    bg = str(bgm).strip()
    c["render"]["bgm"] = str(Path(bg).resolve()) if bg else None
    c["agent"]["llm_enabled"] = bool(llm)
    validate(c)
    return c


def generate_demo() -> dict:
    return generate(SAMPLES)


def probe_info(video: str | Path) -> dict:
    return probe(video)


def preview_roi(video: str | Path, cfg: dict, time_sec: float = 3.0,
                out_png: str | Path | None = None) -> tuple[str, tuple[int, int, int, int]]:
    """Export one frame with the ROI rectangle drawn on it."""
    meta = probe(video)
    x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
    png = (Path(out_png) if out_png else OUTPUTS / f"{Path(video).stem}_roi.png").resolve()
    png.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"drawbox=x={x}:y={y}:w={w}:h={h}:color=red@0.9:t=4,"
          f"drawbox=x={x}:y={y}:w={w}:h={h}:color=yellow@0.6:t=1")
    run_ffmpeg(["-ss", f"{float(time_sec):.3f}", "-i", str(Path(video).resolve()),
                "-frames:v", "1", "-vf", vf, str(png)])
    return str(png), (x, y, w, h)
