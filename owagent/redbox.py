"""L1 redbox channel (CS2): the kill-feed row highlight that follows the VIEW.

CS2 highlights the kill-feed row of the player whose eyes the camera is
behind: when *I* land a kill, my feed row gets a red box - a solid red
fill right after the kill, settling into a thin red stroke around the
row for several seconds. That makes the red box the feed-side twin of
the killcard: it says "the VIEWED player just killed", which combined
with the POV channel ("the viewed player is the target user") and the
killcard delta ("the card count really moved") pins the kill on the
recorder without trusting any name text.

Measured on 1280x960 (4:3) footage:

  * the highlight is a saturated red (R>90, R-max(G,B)>45) that hugs
    the feed row - right-aligned to the screen edge like every row;
  * scene reds (dust2 brick walls) easily fool a raw red mask, but they
    never form a NEAR-FULL-WIDTH red strip at the very right edge of
    the feed ROI across several scanlines - only the row highlight
    does (validated: wall-red frames score 0, all true kill frames
    score 4-111 "strong" scanlines);
  * the box appears 0-1s after the kill and can lag up to ~2.5s when
    the viewer just switched spectate targets, so reads run on a wide
    ladder;
  * the box persists ~5-10s and fades away.

Like the killcard channel this never scans continuously: it only reads
the feed ROI around recall-event groups. Pure pixel work - no OCR, no
model - each read is a frame grab plus numpy morphology (~50ms).
"""
from __future__ import annotations

from pathlib import Path

from .util import ToolError, log


class RedboxNotReady(ToolError):
    """The redbox channel cannot run (missing opencv)."""


# 读数阶梯（相对门控时刻）：红框通常 0-1s 内出现，观战切换时最晚 ~2.5s
READ_LADDER = (-0.6, -0.2, 0.2, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0)
GROUP_SEC = 2.5            # 门控事件聚类窗口（与 killcard 一致）

RED_MIN = 90               # 红色判定：R 下限
RED_DELTA = 45             # R 需高出 G/B 的幅度
RUN_FRAC = 0.055           # 水平连续红条最小长度（相对 ROI 宽，380px→21px 实测）
EDGE_FRAC = 0.079          # 右缘条带宽度（相对 ROI 宽，380px→30px 实测）
STRONG_ROWS = 4            # 右缘近满宽红色的扫描行数下限 → 判定红框
STRENGTH_CAP = 40          # strength = strong_rows / 40（封顶 1.0）


def _strong_rows(img) -> int:
    """Count scanlines whose right-edge strip is nearly full of red.

    The geometry filter is what kills scene-red false positives: brick
    walls bleed red everywhere but never as a right-edge-anchored,
    ~full-width horizontal strip on 4+ consecutive scanlines.
    """
    import cv2
    import numpy as np

    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    red = ((r > RED_MIN) & (r - np.maximum(g, b) > RED_DELTA)).astype(np.uint8)
    h, w = red.shape
    run = max(9, int(w * RUN_FRAC))
    # 只保留水平长条状红色（描边线/实心填充），散斑/文字红被腐蚀掉
    hl = cv2.erode(red, np.ones((1, run), np.uint8))
    if not hl.any():
        return 0
    edge_w = max(12, int(w * EDGE_FRAC))
    right = hl[:, w - edge_w:]
    per_row = right.sum(axis=1)
    return int((per_row >= max(6, int(edge_w * 0.65))).sum())


def _read_feed(video: Path, t: float, roi: tuple[int, int, int, int]):
    """Grab the feed ROI once; return None if the seek fails."""
    from .skull import _grab_frame
    x, y, w, h = roi
    return _grab_frame(video, t, f"crop={w}:{h}:{x}:{y}", w, h)


def _group_gates(gate_events: list[dict], duration: float) -> list[float]:
    """Collapse recall events into group start times (same as killcard)."""
    ts = sorted(e["t"] for e in gate_events)
    groups: list[float] = []
    for t in ts:
        if groups and t - groups[-1] <= GROUP_SEC:
            continue
        groups.append(t)
    return [g for g in groups if g + READ_LADDER[-1] < duration and g + READ_LADDER[0] >= 0]


def detect_redbox(video: str | Path, cfg: dict, meta: dict,
                  gate_events: list[dict]) -> dict:
    """Detect the player-kill red box around recall events (CS2 only).

    Returns {"events", "reads", "boxes"}; events carry source="redbox",
    confirmed=True, kills=1 and strength = scanline evidence (0..1).
    Like the killcard, each event means "the VIEWED player killed here" -
    whether that is the target user is decided by the POV channel.
    """
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    if game != "csgo2":
        return {"events": [], "reads": 0, "boxes": 0}
    try:
        import cv2          # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise RedboxNotReady(f"击杀红框通道需要 opencv（{exc}）。"
                             "安装: pip install -r requirements-ocr.txt") from exc

    # ROI 直接用启发式召回的击杀播报区（红框就在播报行上）
    from .util import even
    roi_cfg = cfg["roi"]
    x = int(roi_cfg["x"] * meta["width"])
    y = int(roi_cfg["y"] * meta["height"])
    w = even(min(int(roi_cfg["w"] * meta["width"]), meta["width"] - x))
    h = even(min(int(roi_cfg["h"] * meta["height"]), meta["height"] - y))
    x = even(x)
    y = even(y)
    roi = (x, y, w, h)

    video = Path(video)
    duration = float(meta["duration"])
    gates = _group_gates(gate_events, duration)
    log(f"CS2 击杀红框识别：播报区 {w}x{h}@({x},{y})，{len(gates)} 个事件组"
        f"（阶梯 {len(READ_LADDER)} 点读数）", "redbox")

    events: list[dict] = []
    reads = 0
    for g in gates:
        best = 0
        for off in READ_LADDER:
            img = _read_feed(video, g + off, roi)
            reads += 1
            if img is None:
                continue
            best = max(best, _strong_rows(img))
        if best >= STRONG_ROWS:
            events.append({
                "t": round(g, 3), "source": "redbox",
                "strength": round(min(best / STRENGTH_CAP, 1.0), 3),
                "confirmed": True, "kills": 1,
            })

    total = sum(e["kills"] for e in events)
    log(f"击杀红框识别完成：{reads} 次读数 → {len(events)} 组视角玩家击杀（共 {total} 杀）",
        "redbox")
    return {"events": events, "reads": reads, "boxes": len(events)}
