"""L1 POV channel (CS2): whose eyes is this footage through?

A CS2 recording is not always the recorder's own view: after dying (or
when watching a GOTV demo) the camera follows OTHER players. Every
"player-private" HUD signal - the kill-feed red box, the bottom-center
kill card - belongs to whoever is being VIEWED, not to the recorder.
Without a POV check the pipeline happily books a spectated teammate's
kills as the target user's.

Two HUD elements give the POV away (measured on 1280x960 footage):

  * the spectate control bar - a thin hint strip at the very bottom
    ("[MOUSE1] 下一名玩家  [CTRL] 总览 ...") that only exists while
    the camera follows another player. Cheap to detect: bright text
    rows in an otherwise empty bottom strip (buy-menu hints sit
    higher and never reach the strip's text density);
  * the bottom-center info card - while spectating it carries the
    VIEWED player's name + their weapon; the recorder's own live HUD
    has no name there at all. Reading the name tells us WHO is viewed,
    which is what the red box / kill card actually belong to.

Pipeline position: after the red box + killcard channels have marked
"the viewed player killed here" events, this channel answers "was the
viewed player the target user?" - events seen through someone else's
eyes are demoted, and whole other-POV stretches are excluded from the
highlight reel.

The target user's name comes from config (ocr.player_name) when given;
otherwise it is voted from the kill-feed killer names at LIVE-POV
confirm moments (the recorder's own kills are the ones confirmed while
no spectate bar is on screen).

Dependencies: rapidocr (namecard / feed reads) - same as the OCR
channel; missing deps degrade gracefully.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .util import NO_WINDOW, ToolError, ffmpeg_bin, log, read_exact


class POVNotReady(ToolError):
    """The POV channel cannot run (missing rapidocr / deps)."""


# 底部观战控制条 ROI（相对坐标，1280x960 实测 x 380-1000 / y 933-959）
BAR_ROI = {"x": 0.297, "y": 0.972, "w": 0.484, "h": 0.027}
# 底部信息卡（观战时显示被观战者名字+武器）
CARD_ROI = {"x": 0.281, "y": 0.885, "w": 0.4375, "h": 0.0938}

BAR_FPS = 2.0          # 控制条扫描帧率（纯像素，便宜）
BAR_MIN_ROWS = 8       # 条带内"亮文字扫描行"数下限 → 疑似观战条
BAR_MIN_BRIGHT = 600   # 条带亮像素总数下限
BAR_PAD = 1.5          # 区间前后外扩秒数
BAR_MERGE_GAP = 10.0   # 观战条短暂消失（被计分板遮挡等）的桥接上限
BAR_MIN_SEC = 1.0      # 短于 1s 的孤立尖峰（菜单残影）不算区间

CARD_SAMPLE_STEP = 4.0  # 观战区间内的名牌采样间隔
ANCHOR_READ_OFFSETS = (0.0, 1.0)   # 锚点时刻的补读 offset

# 名牌/播报里不是玩家名的词（武器、皮肤、状态、拾取提示）
_NOT_NAME = ("刺刀", "匕首", "步枪", "狙击", "手枪", "冲锋", "霰弹", "机炮",
             "AK", "AWP", "M4", "USP", "M9", "M249", "AUG", "SG", "SCAR",
             "GALIL", "FAMAS", "MAC", "MP9", "MP5", "UMP", "P90", "P250",
             "DEAGLE", "沙漠之鹰", "格洛克", "双持", "R8", "NOVA", "XM",
             "MAG", "内格夫", "弹", "甲", "血", "伤", "投掷", "雷", "盾",
             "买", "退款", "购买", "投掷", "总览", "玩家", "聊天", "射线",
             "捡起", "掉落", "StatTrak", "stattrak")
NAME_SIM = 0.60         # 名字模糊匹配阈值（OCR 换字 J1La0Sh1s / JILa0Sh1s）


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _is_name(text: str) -> bool:
    """Leftmost plausible PLAYER name token from the info-card OCR."""
    t = str(text).strip().rstrip("↯>»«×")
    if not (2 <= len(t) <= 24):
        return False
    if any(w.lower() in t.lower() for w in _NOT_NAME):
        return False
    digits = sum(1 for ch in t if ch.isdigit() or ch in ".%|:/_- ")
    if digits / max(len(t), 1) > 0.6:
        return False
    return any("\u4e00" <= ch <= "\u9fff" or ch.isalpha() for ch in t)


def _leftmost_name(result) -> str:
    """The viewed player's name = leftmost name-like OCR box in the card."""
    if result is None or not getattr(result, "txts", None):
        return ""
    boxes = list(result.boxes) if result.boxes is not None else []
    best: tuple[float, str] | None = None
    for box, txt in zip(boxes, result.txts):
        s = str(txt).strip()
        if not _is_name(s):
            continue
        xs = [p[0] for p in box]
        cx = sum(xs) / len(xs)
        if best is None or cx < best[0]:
            best = (cx, s)
    return best[1] if best else ""


def _scan_bar(video: Path, cfg: dict, meta: dict) -> list[tuple[float, float]]:
    """Continuous cheap scan for the spectate control bar → raw intervals."""
    import numpy as np
    from .util import even

    pc = cfg.get("pov", {})
    roi = {**BAR_ROI, **(pc.get("bar") or {})}
    x = even(int(roi["x"] * meta["width"]))
    y = even(int(roi["y"] * meta["height"]))
    w = even(min(int(roi["w"] * meta["width"]), meta["width"] - x))
    h = even(min(int(roi["h"] * meta["height"]), meta["height"] - y))

    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", f"fps={BAR_FPS},crop={w}:{h}:{x}:{y}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    hits: list[float] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          creationflags=NO_WINDOW) as proc:
        assert proc.stdout is not None
        idx = 0
        while True:
            frame = read_exact(proc.stdout, w * h * 3)
            if len(frame) < w * h * 3:
                break
            t = idx / BAR_FPS
            idx += 1
            img = np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3)
            bright = (img.min(axis=2) > 150)
            rows = bright.sum(axis=1)
            if int((rows > 30).sum()) >= BAR_MIN_ROWS and int(bright.sum()) >= BAR_MIN_BRIGHT:
                hits.append(t)
    if not hits:
        return []
    # 单帧命中 → 区间；按 gap 桥接合并
    spans: list[list[float]] = []
    for t in hits:
        if spans and t - spans[-1][1] <= 1.0 / BAR_FPS + 0.1:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo - merged[-1][1] <= BAR_MERGE_GAP:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    out = []
    for lo, hi in merged:
        if hi - lo >= BAR_MIN_SEC:
            out.append((max(0.0, lo - BAR_PAD), hi + BAR_PAD))
    return out


def _read_card(video: Path, t: float, roi: tuple[int, int, int, int], engine) -> str:
    """OCR the info card once; return the viewed player's name ('' = live)."""
    from .skull import _grab_frame
    x, y, w, h = roi
    img = _grab_frame(video, t, f"crop={w}:{h}:{x}:{y}", w, h)
    if img is None:
        return ""
    return _leftmost_name(engine(img))


def _feed_killers_at(video: Path, t: float, roi: tuple[int, int, int, int],
                     engine) -> list[str]:
    """Killer names on the kill feed around time t (owner voting evidence)."""
    from .ocr import _parse_rows
    from .skull import _grab_frame
    x, y, w, h = roi
    names: list[str] = []
    for off in (-0.5, 0.8, 1.8):
        img = _grab_frame(video, t + off, f"crop={w}:{h}:{x}:{y}", w, h)
        if img is None:
            continue
        for killer, _victim, _score in _parse_rows(engine(img), 0):
            if _is_name(killer):
                names.append(killer)
    return names


def _vote_owner(names: list[str]) -> str:
    """Fuzzy-cluster name votes; return the dominant spelling."""
    if not names:
        return ""
    clusters: list[tuple[str, list[str]]] = []   # (rep, members)
    for n in names:
        nn = _norm(n)
        for i, (rep, members) in enumerate(clusters):
            if _similar(nn, _norm(rep)) >= NAME_SIM or nn == _norm(rep):
                members.append(n)
                break
        else:
            clusters.append((n, [n]))
    # 票数并列时取最长名字（更可能是完整读法）
    best = max(clusters, key=lambda c: (len(c[1]), len(c[0])))
    rep, members = best
    counts: dict[str, int] = {}
    for m in members:
        counts[m] = counts.get(m, 0) + 1
    return max(counts, key=counts.get) or rep


def detect_pov(video: str | Path, cfg: dict, meta: dict,
               anchor_events: list[dict]) -> dict:
    """Map the recording's POV: live-own vs spectating-whom (CS2 only).

    `anchor_events` are redbox/killcard "viewed player killed" events.
    Returns {"windows", "anchor_views", "owner", "other_pov", "reads"}:
      * windows   - [{start,end,names}] spectate windows w/ name samples
      * anchor_views - {t: viewed-name-or-''} for every anchor
      * owner     - target user display name (config or voted)
      * other_pov - [(lo,hi)] stretches viewed through someone else's eyes
    """
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    if game != "csgo2":
        return {"windows": [], "anchor_views": {}, "owner": "",
                "other_pov": [], "reads": 0}
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise POVNotReady(f"视角通道需要 rapidocr（{exc}）。"
                          "安装: pip install -r requirements-ocr.txt") from exc

    video = Path(video)
    duration = float(meta["duration"])
    from .util import even
    pc = cfg.get("pov", {})
    card_roi_cfg = {**CARD_ROI, **(pc.get("card") or {})}
    cx = even(int(card_roi_cfg["x"] * meta["width"]))
    cy = even(int(card_roi_cfg["y"] * meta["height"]))
    cw = even(min(int(card_roi_cfg["w"] * meta["width"]), meta["width"] - cx))
    ch = even(min(int(card_roi_cfg["h"] * meta["height"]), meta["height"] - cy))
    card_roi = (cx, cy, cw, ch)

    feed_roi_cfg = cfg["roi"]
    fx = even(int(feed_roi_cfg["x"] * meta["width"]))
    fy = even(int(feed_roi_cfg["y"] * meta["height"]))
    fw = even(min(int(feed_roi_cfg["w"] * meta["width"]), meta["width"] - fx))
    fh = even(min(int(feed_roi_cfg["h"] * meta["height"]), meta["height"] - fy))
    feed_roi = (fx, fy, fw, fh)

    engine = RapidOCR(params={"Global.max_side_len": 480, "Global.text_score": 0.4})
    import logging
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)

    anchors = sorted({round(float(e["t"]), 3) for e in anchor_events if e.get("confirmed")})
    windows_raw = _scan_bar(video, cfg, meta)
    log(f"CS2 视角判定：观战控制条扫描 → {len(windows_raw)} 个候选区间"
        + (f"（{' / '.join(f'{lo:.0f}-{hi:.0f}s' for lo, hi in windows_raw)}" if windows_raw else "")
        + f"；{len(anchors)} 个击杀锚点", "pov")

    reads = 0
    windows: list[dict] = []
    anchor_views: dict[float, str] = {}

    def _card_name(t: float) -> str:
        nonlocal reads
        best = ""
        for off in ANCHOR_READ_OFFSETS:
            tt = t + off
            if not (0 <= tt < duration):
                continue
            reads += 1
            n = _read_card(video, tt, card_roi, engine)
            if n:
                best = n
                break
        return best

    for lo, hi in windows_raw:
        names: list[tuple[float, str]] = []
        t = lo + 0.5
        while t <= hi:
            n = _card_name(t)
            if n:
                names.append((t, n))
            t += CARD_SAMPLE_STEP
        # 锚点落在区间内的：逐个点名（被观战者可能中途切换）
        for a in anchors:
            if lo <= a <= hi:
                n = _card_name(a)
                if n:
                    names.append((a, n))
        names.sort()
        windows.append({"start": round(lo, 2), "end": round(hi, 2),
                        "names": [{"t": round(t, 1), "name": n} for t, n in names]})

    for a in anchors:
        if any(w["start"] <= a <= w["end"] for w in windows):
            n = next((x["name"] for x in
                      next(w for w in windows if w["start"] <= a <= w["end"])["names"]
                      if abs(x["t"] - a) <= 1.0), "")
            anchor_views[a] = n
        else:
            anchor_views[a] = ""    # 无观战条 = 录像者本人视角

    # 窗间补漏：锚点在控制条扫描漏掉的观战段里（条被遮挡/渐隐）。
    # 窗外锚点的单次名牌读数可能是武器皮肤名/拾取提示误读——只有读出的名字
    # 能对上观战区间里见过的玩家（或全程没扫到任何观战条）才采信。
    window_names = [x["name"] for w in windows for x in w["names"]]
    for a in anchors:
        if anchor_views.get(a):
            continue
        n = _card_name(a)
        if not n:
            continue
        if windows and not any(_similar(_norm(n), _norm(wn)) >= 0.5 for wn in window_names):
            continue                     # 与观战玩家对不上：大概率误读，按本人视角处理
        anchor_views[a] = n
        if not any(w["start"] <= a <= w["end"] for w in windows):
            windows.append({"start": round(max(0.0, a - 2.0), 2),
                            "end": round(min(duration, a + 2.0), 2),
                            "names": [{"t": round(a, 1), "name": n}]})
        else:
            w = next(w for w in windows if w["start"] <= a <= w["end"])
            w["names"].append({"t": round(a, 1), "name": n})
            w["names"].sort(key=lambda x: x["t"])

    # 目标用户名：配置优先；否则用"本人视角锚点"的播报击杀者名投票，
    # 再退回全锚点投票（全观战录制：被反复观看的通常就是主角）
    owner = str(cfg.get("ocr", {}).get("player_name") or "").strip()
    live_anchors = [a for a in anchors if not anchor_views.get(a)]
    if not owner:
        votes: list[str] = []
        for a in live_anchors:
            reads += 1
            votes.extend(_feed_killers_at(video, a, feed_roi, engine))
        if not votes:
            for a in anchors:
                reads += 1
                votes.extend(_feed_killers_at(video, a, feed_roi, engine))
        owner = _vote_owner(votes)
        if owner:
            log(f"视角判定：{len(live_anchors)} 个本人视角锚点投票 → 目标用户「{owner}」", "pov")
    else:
        log(f"视角判定：使用配置玩家名「{owner}」", "pov")

    # 观战他人区间：区间内有名字读数、且没有任何一个匹配目标用户
    other_pov: list[tuple[float, float]] = []
    for w in windows:
        names = [x["name"] for x in w["names"]]
        if not names:
            continue                     # 无名读数：多为购买菜单残影，不算
        if owner and any(_similar(_norm(n), _norm(owner)) >= NAME_SIM for n in names):
            continue                     # 观战的就是目标用户本人（GOTV 回放）
        other_pov.append((w["start"], w["end"]))

    # 合并相邻区间
    merged: list[list[float]] = []
    for lo, hi in sorted(other_pov):
        if merged and lo <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    other_pov = [(lo, hi) for lo, hi in merged]

    spect = sum(hi - lo for lo, hi in other_pov)
    log(f"视角判定完成：{len(windows)} 个观战区间，剔除观战他人视角 "
        f"{len(other_pov)} 段共 {spect:.0f}s（目标用户={owner or '未知'}）", "pov")
    return {"windows": windows, "anchor_views": {str(k): v for k, v in anchor_views.items()},
            "owner": owner, "other_pov": other_pov, "reads": reads}


def demote_spectated(events: list[dict], pov: dict) -> int:
    """Demote confirmed kills seen through someone else's POV (in place).

    Returns the number of demoted events. An event stays confirmed only
    if it happened in the recorder's own view, or while spectating the
    target user themself.
    """
    owner = pov.get("owner") or ""
    views = pov.get("anchor_views") or {}
    demoted = 0
    for e in events:
        if not e.get("confirmed"):
            continue
        view = views.get(str(round(float(e["t"]), 3)), "")
        if not view:
            continue                     # 本人视角：红框/卡牌就是本人的
        if owner and _similar(_norm(view), _norm(owner)) >= NAME_SIM:
            continue                     # 观战目标用户本人（GOTV）
        e["confirmed"] = False
        e["spectate"] = view
        demoted += 1
    return demoted
