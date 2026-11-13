"""L1 visual recall: cheap, dependency-free event detection on the kill-feed ROI.

ffmpeg decodes only a tiny grayscale ROI at 8fps; Python stdlib computes three
signals (frame difference, newly appeared HUD ink, absolute HUD ink level).
Only an *increase* in HUD ink above threshold counts as an event, so fade-outs,
sustained banners and camera motion do not create duplicates.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .config import pixel_roi
from .util import NO_WINDOW, ffmpeg_bin, log, read_exact

HIGHLIGHT_PIXEL = 170  # grayscale value above which a pixel counts as "HUD ink"


def detect_visual(video: str | Path, cfg: dict, meta: dict) -> dict:
    x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
    s = cfg["sample"]
    d = cfg["detect"]
    fw, fh, fps = int(s["width"]), int(s["height"]), float(s["fps"])
    frame_size = fw * fh  # gray = 1 byte per pixel

    vf = f"crop={w}:{h}:{x}:{y},fps={fps},scale={fw}:{fh},format=gray"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", vf, "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    log(f"ROI 像素框 x={x} y={y} w={w} h={h} → 采样 {fw}x{fh} @ {fps}fps", "detect")

    events: list[dict] = []
    signals: list[dict] = []
    prev: bytes | None = None
    prev_ink = 0.0
    last_event_t = -1e9
    idx = 0

    with tempfile.TemporaryFile(mode="w+b") as errfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile,
                                creationflags=NO_WINDOW)
        assert proc.stdout is not None
        try:
            while True:
                frame = read_exact(proc.stdout, frame_size)
                if len(frame) < frame_size:
                    break
                t = idx / fps
                mean = sum(frame) / frame_size
                ink = sum(1 for v in frame if v > HIGHLIGHT_PIXEL) / frame_size
                diff = (sum(abs(a - b) for a, b in zip(frame, prev)) / frame_size) if prev else 0.0
                rise = max(0.0, ink - prev_ink)  # newly appeared HUD ink = a fresh kill-feed row

                nd = min(diff / float(d["diff_gain"]), 1.0)
                nr = min(rise / float(d["rise_gain"]), 1.0)
                ni = min(ink / float(d["ink_gain"]), 1.0)
                score = (float(d["w_diff"]) * nd + float(d["w_rise"]) * nr
                         + float(d["w_ink"]) * ni)

                # Only the *appearance* of HUD ink is an event. Fade-outs,
                # sustained banners and background motion must not trigger it.
                if (rise >= float(d["min_rise"]) and score >= float(d["threshold"])
                        and (t - last_event_t) >= float(d["refractory_sec"])):
                    events.append({
                        "t": round(t, 3), "source": "visual",
                        "strength": round(min(score, 1.0), 4),
                        "diff": round(diff, 2), "ink": round(ink, 4),
                        "ink_rise": round(rise, 4),
                    })
                    last_event_t = t

                signals.append({"t": round(t, 3), "diff": round(diff, 3),
                                "ink": round(ink, 5), "ink_rise": round(rise, 5),
                                "level": round(mean, 3), "score": round(score, 4)})
                prev = frame
                prev_ink = ink
                idx += 1
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-1200:]
            log(f"视觉解码失败: {detail}", "warn")

    log(f"扫描 {idx} 帧 / {idx / fps:.1f}s，召回 {len(events)} 个视觉事件", "detect")
    return {"events": events, "signals": signals, "frames": idx,
            "roi_px": [x, y, w, h], "fps": fps}
