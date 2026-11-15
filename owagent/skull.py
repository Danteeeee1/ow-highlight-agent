"""L1 skull channel (Overwatch): the screen-center kill-confirm skull.

OW2 shows a small red skull near the crosshair when *the recorder* lands an
elimination - assists included, and only the killer sees it. That makes the
skull the single most player-specific HUD signal in the game: the kill feed
shows everyone's kills, but the center skull means "MY kill".

Detection signature (measured on 1920x1080 footage, scales with height):

  * a solid red blob (HSV hue <12 / >168, sat>70, val>90), 40-150px,
    within ~130px of the crosshair (screen center);
  * the skull's two eye sockets show up as 8-40px HOLES inside the blob
    (contour hierarchy children), side by side: dx 20-65px, dy <16px,
    similar sizes. Explosion noise and enemy outlines have random holes
    that rarely match this geometry;
  * lifetime 0.15-0.5s (the icon fades out) - blobs persisting >1.3s are
    static scenery (glowing robot eyes etc.) and are rejected;
  * kill-confirm flash often merges the skull with the red X hit marker
    into one blob - size limits are generous for that.

Optional feed confirmation: OCR the kill feed at t-1.5 / t+1.2 - a NEW
(killer, victim) row appearing after the skull upgrades the event to a
confirmed player kill and attaches the names. Unconfirmed candidates keep
a weaker strength (possibly assist elims, possibly noise).

Dependencies: opencv-python (cv2) - already required by the OCR channel
(requirements-ocr.txt). Missing deps degrade gracefully.
"""
from __future__ import annotations

import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

from .util import NO_WINDOW, ToolError, ffmpeg_bin, log, read_exact


class SkullNotReady(ToolError):
    """The skull channel cannot run (missing opencv)."""


# --- geometry at 1080p reference (scales with video height) ---
REF_H = 1080
CROP_HALF_W = 170          # crop half-size around the crosshair
CROP_HALF_H = 150
ZONE_DX = 130              # skull center tolerance vs crosshair
ZONE_DY = 130
BLOB_MIN, BLOB_MAX = 40, 150
HOLE_MIN, HOLE_MAX = 8, 40
EYE_DX_MIN, EYE_DX_MAX = 20, 65
EYE_DY_MAX = 16
STATIC_LIFE = 1.3          # blobs alive longer than this are scenery
TRACK_GAP = 0.35           # hits farther apart than this start a new track
CONFIRM_BEFORE = 1.5       # feed OCR offset before the skull
CONFIRM_AFTER = 1.2        # feed OCR offset after the skull


def _frame_hits(img, scale: float) -> list[dict]:
    """All skull-shaped blobs in one frame (crosshair zone only).

    `img` is the cropped BGR->RGB center region; `scale` = height/1080.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h < 12) | (h > 168)) & (s > 70) & (v > 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, hier = cv2.findContours(mc, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []

    ch, cw = img.shape[:2]
    cxr, cyr = cw / 2, ch / 2                    # crosshair in crop coords
    bmin, bmax = BLOB_MIN * scale, BLOB_MAX * scale
    hmin, hmax = HOLE_MIN * scale, HOLE_MAX * scale
    out = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue                              # outer contours only
        x, y, w, h_ = cv2.boundingRect(c)
        if not (bmin <= w <= bmax and bmin <= h_ <= bmax):
            continue
        if not (0.5 <= w / max(h_, 1) <= 1.8):
            continue
        cx, cy = x + w / 2, y + h_ / 2
        if abs(cx - cxr) > ZONE_DX * scale or abs(cy - cyr) > ZONE_DY * scale:
            continue
        holes = []
        for j, hc in enumerate(cnts):
            if hier[0][j][3] != i:
                continue
            hx, hy, hw, hh = cv2.boundingRect(hc)
            if hmin <= hw <= hmax and hmin <= hh <= hmax and cv2.contourArea(hc) >= 40 * scale * scale:
                holes.append((hx + hw / 2, hy + hh / 2, hw, hh))
        for a, b in combinations(holes, 2):
            dx, dy = abs(b[0] - a[0]), abs(a[1] - b[1])
            if dy < EYE_DY_MAX * scale and EYE_DX_MIN * scale <= dx <= EYE_DX_MAX * scale:
                sim = min(a[2] * a[3], b[2] * b[3]) / max(a[2] * a[3], b[2] * b[3])
                area = cv2.contourArea(c)
                hull = cv2.contourArea(cv2.convexHull(c))
                out.append({
                    "cx": cx, "cy": cy, "w": w, "h": h_,
                    "eye_sim": sim,
                    "fill": area / max(w * h_, 1),
                    "solidity": area / max(hull, 1),
                })
                break
    return out


def _shape_score(hit: dict) -> float:
    """0-1 confidence that a hit is the kill skull (not combat noise)."""
    score = hit["eye_sim"] * 0.45
    if hit["fill"] >= 0.55:
        score += 0.2
    elif hit["fill"] >= 0.4:
        score += 0.1
    if hit["solidity"] >= 0.8:
        score += 0.2
    elif hit["solidity"] >= 0.6:
        score += 0.1
    # a clean skull is roughly square
    if 0.75 <= hit["w"] / max(hit["h"], 1) <= 1.35:
        score += 0.15
    return min(score, 1.0)


def _grab_frame(video: Path, t: float, vf: str, w: int, h: int):
    """One raw RGB frame at time t (None if seek fails)."""
    import numpy as np
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW).stdout
    if len(out) < w * h * 3:
        return None
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3)


def _feed_rows_at(video: Path, t: float, ocr_engine, feed_roi, game: str):
    """OCR the kill feed once; return [(killer, victim)] rows."""
    from .ocr import _parse_rows
    fx, fy, fw, fh = feed_roi
    img = _grab_frame(video, t, f"crop={fw}:{fh}:{fx}:{fy}", fw, fh)
    if img is None:
        return []
    return [(killer, victim)
            for killer, victim, _score in _parse_rows(ocr_engine(img), 0)]


def _norm_pair(killer: str, victim: str) -> tuple[str, str]:
    from .ocr import _norm_name
    return _norm_name(killer), _norm_name(victim)


def detect_skull(video: str | Path, cfg: dict, meta: dict) -> dict:
    """Scan for the OW kill-confirm skull; events carry source="skull".

    Returns {"events", "frames", "fps", "candidates", "confirmed"}.
    Raises SkullNotReady when opencv is missing (pipeline skips the channel).
    """
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    if game != "overwatch":
        return {"events": [], "frames": 0, "fps": 0, "candidates": 0, "confirmed": 0}
    try:
        import cv2          # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise SkullNotReady(
            f"骷髅头通道需要 opencv（{exc}）。"
            "安装: pip install -r requirements-ocr.txt") from exc

    sk = cfg.get("skull", {})
    fps = float(sk.get("fps", 16))
    scale = meta["height"] / REF_H
    cw, ch = int(CROP_HALF_W * scale * 2), int(CROP_HALF_H * scale * 2)
    x0 = int(meta["width"] / 2 - CROP_HALF_W * scale)
    y0 = int(meta["height"] / 2 - CROP_HALF_H * scale)
    vf = f"fps={fps},crop={cw}:{ch}:{max(x0, 0)}:{max(y0, 0)}"

    log(f"OW 击杀骷髅头识别：准星区 {cw}x{ch} @ {fps}fps，"
        f"眼窝签名（双孔 dy<{EYE_DY_MAX*scale:.0f}px dx {EYE_DX_MIN*scale:.0f}-{EYE_DX_MAX*scale:.0f}px）", "skull")

    frame_size = cw * ch * 3
    hits: list[tuple[float, dict]] = []
    with tempfile.TemporaryFile(mode="w+b") as errfile:
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
               "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24",
               "-f", "rawvideo", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile,
                                creationflags=NO_WINDOW)
        assert proc.stdout is not None
        idx = 0
        try:
            while True:
                frame = read_exact(proc.stdout, frame_size)
                if len(frame) < frame_size:
                    break
                t = idx / fps
                idx += 1
                img = np.frombuffer(frame, dtype=np.uint8).reshape(ch, cw, 3)
                for hit in _frame_hits(img, scale):
                    hits.append((t, hit))
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-800:]
            log(f"骷髅头通道解码失败: {detail}", "warn")

    # ---- group hits into tracks (position continuity), then score ----
    tracks: list[dict] = []
    for t, hit in hits:
        for tr in reversed(tracks):
            if t - tr["last"] > TRACK_GAP:
                continue
            if abs(hit["cx"] - tr["cx"]) <= 70 * scale and abs(hit["cy"] - tr["cy"]) <= 70 * scale:
                tr["pts"].append((t, hit))
                tr["cx"], tr["cy"], tr["last"] = hit["cx"], hit["cy"], t
                break
        else:
            tracks.append({"pts": [(t, hit)], "cx": hit["cx"],
                           "cy": hit["cy"], "last": t})

    # time-merge nearby tracks (blob jumps when merging with explosion flash)
    merged: list[dict] = []
    for tr in sorted(tracks, key=lambda z: z["pts"][0][0]):
        if merged and tr["pts"][0][0] - merged[-1]["last"] <= TRACK_GAP:
            merged[-1]["pts"].extend(tr["pts"])
            merged[-1]["last"] = max(merged[-1]["last"], tr["last"])
        else:
            merged.append(tr)

    events: list[dict] = []
    n_static = 0
    for tr in merged:
        pts = tr["pts"]
        ts = [p[0] for p in pts]
        life = max(ts) - min(ts)
        if life > STATIC_LIFE:
            n_static += 1                      # glowing scenery, HUD junk
            continue
        best = max(pts, key=lambda p: _shape_score(p[1]))
        score = _shape_score(best[1])
        if len(pts) >= 2:
            score = min(1.0, score + 0.12)      # multi-frame persistence
        events.append({
            "t": round(min(ts), 3),
            "source": "skull",
            "strength": round(max(0.3, score), 3),
            "skull_score": round(score, 3),
            "skull_frames": len(pts),
            "skull_life": round(life, 2),
        })
    events.sort(key=lambda e: e["t"])

    # ---- optional kill-feed confirmation (needs rapidocr) ----
    confirmed = 0
    if sk.get("confirm_feed", True) and events:
        try:
            from rapidocr import RapidOCR
            from .config import pixel_roi
            engine = RapidOCR(params={"Global.max_side_len": 640,
                                      "Global.text_score": 0.4})
            # 必须在引擎创建之后再压级别：RapidOCR 初始化会重置 logger（见 ocr.py）
            import logging
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            fx, fy, fw, fh = pixel_roi(cfg, meta["width"], meta["height"])
            feed_roi = (fx, fy, fw, fh)
        except ImportError:
            engine = None
            log("骷髅头通道：rapidocr 不可用，跳过击杀播报确认（仅形状置信度）", "warn")
        if engine is not None:
            for ev in events:
                t = ev["t"]
                if t - CONFIRM_BEFORE < 0 or t + CONFIRM_AFTER > meta["duration"]:
                    continue
                before = {p for p in (_norm_pair(k, v)
                                      for k, v in _feed_rows_at(video, t - CONFIRM_BEFORE, engine, feed_roi, game))}
                if before is None:
                    before = set()
                # 播报行出现后停留 ~9s：读两个时间点防 OCR 偶发漏读
                new_rows: list[tuple[str, str]] = []
                for dt in (CONFIRM_AFTER, CONFIRM_AFTER + 1.3):
                    if t + dt > meta["duration"]:
                        break
                    for k, v in _feed_rows_at(video, t + dt, engine, feed_roi, game):
                        if _norm_pair(k, v) not in before:
                            new_rows.append((k, v))
                    if new_rows:
                        break
                if new_rows:
                    # 播报新行从顶部插入：最上面的新行最可能是本次击杀
                    killer, victim = new_rows[0]
                    ev["strength"] = 0.95
                    ev["killer"], ev["victim"] = killer, victim
                    ev["confirmed"] = True
                    confirmed += 1

    log(f"骷髅头扫描完成：{len(hits)} 帧命中 → {len(events)} 个候选"
        f"（剔除 {n_static} 个静态物，{confirmed} 个经击杀播报确认为玩家击杀）", "skull")
    return {"events": events, "frames": int(len(hits) and hits[-1][0] * fps + 1 or 0),
            "fps": fps, "candidates": len(events), "confirmed": confirmed}
