"""Optional L1 audio recall: RMS envelope peaks from a mono PCM stream."""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

from .util import NO_WINDOW, ffmpeg_bin, log, read_exact


def detect_audio(video: str | Path, cfg: dict, has_audio: bool = True) -> dict:
    a = cfg["audio"]
    if not a["enabled"]:
        return {"events": [], "rms": [], "skipped": "已在配置/命令行中禁用"}
    if not has_audio:
        return {"events": [], "rms": [], "skipped": "源视频无音轨，已自动跳过"}
    rate = int(a["sample_rate"])
    win = float(a["window_sec"])
    chunk_size = max(1, int(rate * win)) * 2  # s16le = 2 bytes/sample

    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(rate), "-acodec", "pcm_s16le",
           "-f", "s16le", "-"]
    events: list[dict] = []
    series: list[dict] = []
    baseline: float | None = None
    last_t = -1e9
    idx = 0

    with tempfile.TemporaryFile(mode="w+b") as errfile:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile,
                                    creationflags=NO_WINDOW)
        except OSError as exc:
            return {"events": [], "rms": [], "skipped": f"ffmpeg 启动失败: {exc}"}
        assert proc.stdout is not None
        try:
            while True:
                raw = read_exact(proc.stdout, chunk_size)
                if len(raw) < 2:
                    break
                if len(raw) % 2:
                    raw = raw[:-1]
                samples = array("h")
                samples.frombytes(raw)
                if sys.byteorder == "big":
                    samples.byteswap()
                n = len(samples)
                rms = math.sqrt(sum(v * v for v in samples) / n) / 32768.0 if n else 0.0
                t = idx * win
                if baseline is None:
                    baseline = rms
                ratio = rms / max(baseline, 1e-6)
                if (rms >= float(a["min_rms"]) and ratio >= float(a["peak_ratio"])
                        and (t - last_t) >= float(a["refractory_sec"])):
                    events.append({"t": round(t, 3), "source": "audio",
                                   "strength": round(min(ratio / (float(a["peak_ratio"]) * 2), 1.0), 4),
                                   "rms": round(rms, 5)})
                    last_t = t
                baseline += float(a["baseline_alpha"]) * (rms - baseline)
                series.append({"t": round(t, 3), "rms": round(rms, 6)})
                idx += 1
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-800:]
            return {"events": [], "rms": [], "skipped": f"无音轨或解码失败: {detail[:200]}"}

    log(f"音频 {idx * win:.1f}s，召回 {len(events)} 个峰值事件", "detect")
    return {"events": events, "rms": series, "skipped": None}
