"""Render the selected timeline into an MP4 with ffmpeg (plus .srt captions)."""
from __future__ import annotations

import shutil
from pathlib import Path

from .util import ToolError, fmt_time, log, run_ffmpeg


def make_srt(selected: list[dict]) -> str:
    lines: list[str] = []
    cursor = 0.0
    for seg in selected:
        a, b = cursor, cursor + seg["duration"]
        lines += [str(len(lines) // 4 + 1),
                  f"{fmt_time(a, srt=True)} --> {fmt_time(b, srt=True)}",
                  seg["caption"], ""]
        cursor = b
    return "\n".join(lines)


def render(video: str | Path, selected: list[dict], out_path: str | Path,
           work_dir: str | Path, cfg: dict, meta: dict) -> dict:
    if not selected:
        raise ToolError("没有入选片段，无法渲染。请降低 --threshold 或看 signals.csv 调参。")
    work_dir = Path(work_dir)
    parts_dir = work_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = cfg["render"]
    # width/height/fps = 0 表示"跟随源"：不做降采样降帧率，成片保持录像
    # 原始画质（1080p60 源出 1080p60 成片）。显式给正数则按指定值缩放。
    src_w, src_h = int(meta.get("width") or 0), int(meta.get("height") or 0)
    src_fps = float(meta.get("fps") or 0)
    w = int(r["width"]) or (src_w or 1280)
    h = int(r["height"]) or (src_h or 720)
    fps = int(r["fps"]) or int(round(src_fps)) or 30
    # yuv420p 要求偶数尺寸
    w, h = w - w % 2, h - h % 2
    same_size = src_w > 0 and src_h > 0 and (w, h) == (src_w, src_h)
    if r["vertical"]:
        vf = f"crop=ih*9/16:ih,scale=1080:1920,fps={fps},setsar=1"
    elif same_size:
        # 与源同分辨率：跳过 scale/pad，避免任何重采样模糊
        vf = f"fps={fps},setsar=1"
    else:
        vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1")

    # 烧字幕会 concat 后再编码一代：中间 part 用近无损档（crf 10），
    # 最终画质只由烧录阶段的 crf 决定，全程只有一次可见的有损压缩。
    final_crf = int(r["crf"])
    part_crf = 10 if r["burn_subtitles"] else final_crf

    has_audio = bool(meta.get("has_audio"))
    part_files: list[Path] = []
    for i, seg in enumerate(selected, 1):
        part = parts_dir / f"part_{i:03d}.mp4"
        dur = seg["end"] - seg["start"]
        if has_audio:
            cmd = ["-ss", f"{seg['start']:.3f}", "-i", str(video),
                   "-map", "0:v:0", "-map", "0:a:0?"]
        else:
            cmd = ["-ss", f"{seg['start']:.3f}", "-i", str(video),
                   "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                   "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-t", f"{dur:.3f}", "-vf", vf,
                "-c:v", "libx264", "-preset", str(r["preset"]), "-crf", str(part_crf),
                "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-avoid_negative_ts", "make_zero", str(part)]
        run_ffmpeg(cmd)
        part_files.append(part)
        log(f"切片 {i}/{len(selected)}  {fmt_time(seg['start'])} → {fmt_time(seg['end'])}", "render")

    concat_list = (parts_dir / "concat.txt").resolve()
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in part_files), encoding="utf-8")
    raw_out = work_dir / "highlights_raw.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy", "-movflags", "+faststart", str(raw_out)], cwd=str(parts_dir))

    staged = raw_out
    burn_ok = False
    if r["burn_subtitles"]:
        srt_path = work_dir / "captions.srt"
        staged_burn = work_dir / "highlights_burn.mp4"
        try:
            run_ffmpeg(["-i", str(raw_out), "-vf", "subtitles=captions.srt",
                        "-c:v", "libx264", "-preset", str(r["preset"]), "-crf", str(final_crf),
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(staged_burn)],
                       cwd=str(work_dir))
            staged = staged_burn
            burn_ok = True
            log("字幕已烧录", "render")
        except ToolError as exc:
            log(f"字幕烧录失败，已跳过（不影响成片）: {str(exc)[-200:]}", "warn")
            if part_crf != final_crf:
                # parts 是近无损中间档（体积大）：回落转码到最终画质档
                try:
                    staged_fallback = work_dir / "highlights_fallback.mp4"
                    run_ffmpeg(["-i", str(raw_out), "-c:v", "libx264",
                                "-preset", str(r["preset"]), "-crf", str(final_crf),
                                "-pix_fmt", "yuv420p", "-c:a", "copy",
                                str(staged_fallback)])
                    staged = staged_fallback
                except ToolError:
                    pass  # 转码也失败：直接交付近无损中间件（画质无损，只是文件大）

    bgm = r.get("bgm")
    if bgm:
        bgm_path = Path(bgm)
        if not bgm_path.is_file():
            log(f"BGM 不存在，已跳过: {bgm_path}", "warn")
            shutil.copyfile(staged, out_path)
        else:
            total = sum(s["duration"] for s in selected)
            run_ffmpeg(["-i", str(staged), "-stream_loop", "-1", "-i", str(bgm_path),
                        "-t", f"{total:.3f}", "-filter_complex",
                        f"[1:a]volume={float(r['bgm_volume']):.3f}[b];"
                        "[0:a][b]amix=inputs=2:duration=first:dropout_transition=2[a]",
                        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path)])
            log("BGM 已混合", "render")
    else:
        shutil.copyfile(staged, out_path)

    log(f"成片已输出: {out_path}", "render")
    return {"output": str(out_path), "parts": [str(p) for p in part_files],
            "raw": str(raw_out), "subtitles_burned": burn_ok,
            "resolution": "1080x1920" if r["vertical"] else f"{w}x{h}"}
