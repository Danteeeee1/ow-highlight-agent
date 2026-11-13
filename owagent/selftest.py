"""Generate a synthetic OW-like clip so the whole pipeline can be tested without a real recording.

Produces a 30s 320x180 video with 8 simulated kill-feed flashes and matching
audio spikes, plus a ground-truth JSON. Built with stdlib only (wave + rawvideo).
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path

from .util import NO_WINDOW, ToolError, ffmpeg_bin, log

W, H, FPS, DURATION = 320, 180, 15, 30
EVENTS = [3.0, 4.0, 5.0, 14.0, 15.0, 25.0, 26.0, 27.0]
FLASH_SEC = 0.45
ROI_X, ROI_Y, ROI_W, ROI_H = 230, 8, 86, 38


def _make_audio(path: Path) -> None:
    sr = 16000
    n_total = sr * DURATION
    buf = array("h", [0]) * n_total
    for n in range(n_total):
        buf[n] = int(140 * math.sin(2 * math.pi * 90 * n / sr))
    for e in EVENTS:
        a = max(0, int((e - 0.05) * sr))
        b = min(n_total, int((e + 0.35) * sr))
        for n in range(a, b):
            buf[n] = int(10500 * math.sin(2 * math.pi * 660 * n / sr))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(buf.tobytes())


def generate(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "demo_match.mp4"
    truth = out_dir / "demo_truth.json"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "demo.wav"
        _make_audio(wav)

        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", str(wav),
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-ar", "44100", "-shortest", "-movflags", "+faststart", str(video)]
        base = bytearray(bytes((18, 26, 38)) * (W * H))
        with tempfile.TemporaryFile(mode="w+b") as errfile:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errfile,
                                    creationflags=NO_WINDOW)
            assert proc.stdin is not None
            try:
                for i in range(FPS * DURATION):
                    t = i / FPS
                    frame = bytearray(base)
                    col = (i * 3) % 120 + 10  # background motion, outside the ROI
                    for yy in range(20, 160, 4):
                        off = (yy * W + col) * 3
                        frame[off:off + 3] = b"\x50\x60\x70"
                    if any(e <= t < e + FLASH_SEC for e in EVENTS):
                        for row in range(3):
                            y0 = ROI_Y + 3 + row * 11
                            for yy in range(y0, min(y0 + 8, ROI_Y + ROI_H)):
                                off = (yy * W + ROI_X) * 3
                                frame[off:off + ROI_W * 3] = b"\xf2\xf2\xf0" * ROI_W
                        for yy in range(ROI_Y + 4, ROI_Y + 12):
                            off = (yy * W + ROI_X + 4) * 3
                            frame[off:off + 24] = b"\xe0\x40\x40" * 8
                    proc.stdin.write(frame)
                proc.stdin.close()
                rc = proc.wait()
            except (BrokenPipeError, OSError):
                proc.stdin.close()
                rc = proc.wait()
            if rc != 0:
                errfile.seek(0)
                raise ToolError("合成视频生成失败: " + errfile.read().decode("utf-8", "replace")[-1200:])

    truth.write_text(json.dumps({
        "note": "合成测试录像的击杀播报真值时间戳（秒）",
        "duration": DURATION, "fps": FPS, "resolution": [W, H],
        "roi_px": [ROI_X, ROI_Y, ROI_W, ROI_H],
        "events": EVENTS
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已生成合成测试录像 {video}（{DURATION}s，{len(EVENTS)} 次模拟击杀播报）", "demo")
    return {"video": str(video), "truth": str(truth), "events": EVENTS}


def compare(detected: list[dict], truth_events: list[float], tol: float = 1.0) -> dict:
    """Recall/precision of detected visual events against ground truth."""
    hits = 0
    matched = set()
    for det in detected:
        for i, t in enumerate(truth_events):
            if i not in matched and abs(det["t"] - t) <= tol:
                matched.add(i)
                hits += 1
                break
    recall = hits / len(truth_events) if truth_events else 0.0
    precision = hits / len(detected) if detected else 0.0
    return {"truth": len(truth_events), "detected": len(detected),
            "matched": hits, "recall": round(recall, 3), "precision": round(precision, 3)}
