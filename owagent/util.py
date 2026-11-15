"""Shared helpers: locating ffmpeg, running commands, probing media, logging."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class ToolError(RuntimeError):
    """Raised when an external tool is missing or fails."""


# Windows 下无控制台进程（pythonw 启动的 GUI）派生 ffmpeg/ffprobe 子进程时，
# 默认会为每个子进程弹一个控制台黑窗。CREATE_NO_WINDOW 阻止弹窗；
# 非 Windows 平台传 0 即可（subprocess 接受 0）。
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def log(msg: str, stage: str = "info") -> None:
    print(f"[{stage}] {msg}", flush=True)


def _locate(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if sys.platform == "win32":
        # Common install locations on Windows (winget / chocolatey / manual).
        patterns = [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
            r"C:\ProgramData\chocolatey\bin",
            r"C:\ffmpeg\bin",
        ]
        for root in patterns:
            base = Path(root)
            if base.exists():
                for candidate in base.rglob(f"{name}.exe"):
                    return str(candidate)
    raise ToolError(
        f"找不到 {name}。请先安装 FFmpeg 并确保它在 PATH 中。\n"
        "  Windows : winget install --id Gyan.FFmpeg -e  (然后重开终端)\n"
        "  macOS   : brew install ffmpeg\n"
        "  Ubuntu  : sudo apt install ffmpeg\n"
        f"  或把 {name}.exe 的完整路径加入 PATH 后重试。"
    )


def ffmpeg_bin() -> str:
    return _locate("ffmpeg")


def ffprobe_bin() -> str:
    return _locate("ffprobe")


def run(cmd: list[str], *, stdin=None, stdout=None, stderr=None, check: bool = True):
    """Run a command, capture diagnostics on failure."""
    proc = subprocess.run(cmd, input=stdin, stdout=stdout, stderr=stderr,
                          creationflags=NO_WINDOW)
    if check and proc.returncode != 0:
        raise ToolError(f"命令失败 (exit {proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def run_ffmpeg(args: list[str], *, cwd: str | None = None, capture: bool = False):
    """Run ffmpeg with -y and error-only logging; surface stderr on failure."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"] + [str(a) for a in args]
    with tempfile.TemporaryFile(mode="w+b") as errfile:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                              stderr=errfile, cwd=cwd, creationflags=NO_WINDOW)
        if proc.returncode != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-2000:]
            raise ToolError(f"FFmpeg 失败 (exit {proc.returncode}):\n{' '.join(cmd)}\n{detail}")
        return proc.stdout if capture else b""


def probe(path: str | Path) -> dict:
    """Return {duration, width, height, has_audio, fps} for a media file."""
    cmd = [ffprobe_bin(), "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,width,height,r_frame_rate",
           "-of", "json", str(path)]
    out = subprocess.check_output(cmd, text=True, creationflags=NO_WINDOW)
    data = json.loads(out)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    width = height = 0
    has_audio = False
    fps = 0.0
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not width:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            rate = st.get("r_frame_rate", "0/1")
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
        elif st.get("codec_type") == "audio":
            has_audio = True
    if not width or duration <= 0:
        raise ToolError(f"无法读取视频信息或时长为 0: {path}")
    return {"duration": duration, "width": width, "height": height,
            "has_audio": has_audio, "fps": fps}


def fmt_time(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a pipe (or until EOF)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def even(v: int) -> int:
    """yuv420p crop needs even coordinates/sizes."""
    return max(2, int(v) // 2 * 2)
