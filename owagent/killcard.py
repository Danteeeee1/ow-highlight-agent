"""L1 killcard channel (CS2): the bottom-center round-kill card stack.

CS2 shows a small fan of cards just above the rank badge at the bottom
center of the screen - one card per kill the RECORDER lands this round,
the newest card carrying the current round kill count. Like the OW skull,
this display is exclusive to the player: the kill feed shows everyone's
kills, the card stack only counts MINE. On top of that the digit itself
IS the round streak - no inference needed.

Measured on 1280x960 (4:3) footage:

  * the stack sits at ~x 0.46-0.57 / y 0.86-0.95 (relative), above the
    always-visible round badge;
  * the fan grows to the RIGHT - card k's digit sits ~17px (source) right
    of card k-1, so the RIGHTMOST readable digit is the current count;
  * every kill pops a fresh card with a bright animation during which the
    digit OCRs at conf 0.97-1.0; the settled stack is tilted and its
    digits flicker in/out of readability across frames (misreads "7"/"WU");
  * the pop is readable from kill-time to ~+0.8s, BUT the heuristic gate
    (kill-feed ink rise) lags the kill by ~0.5s, so the readable window
    often ENDS before gate+0.4 - reads must start BEFORE the gate; some
    kills' cards only become readable ~+2s after the gate - reads must
    also extend well past it;
  * the stack persists for the whole round, then disappears at round end;
    a long stretch of unreadable gates therefore means "round over" and
    the previous count must be discarded (stale reset).

The channel never scans continuously. Every kill - ours included - adds
a kill-feed row / medal, which the other recall channels already detect.
For each recall-event group we read the card on a LADDER of offsets
around the gate and take the maximum count seen; a monotonic state
machine across gates turns count changes into recorder-kill events:

    v == prev        -> someone else's kill / HUD noise (digit unmoved)
    v >  prev        -> recorder landed (v - prev) kills  [cap: v <= 5]
    v <  prev        -> round reset; ladder shows how many kills into the
                        new round (progressive readings 1,2,3 = triple)
    no card for >8s  -> round over, prev discarded (stale)

Dependencies: rapidocr (requirements-ocr.txt) - same as the OCR channel;
missing deps degrade gracefully (KillcardNotReady, pipeline skips).
"""
from __future__ import annotations

from pathlib import Path

from .util import ToolError, log


class KillcardNotReady(ToolError):
    """The killcard channel cannot run (missing rapidocr / deps)."""


# 卡牌区相对坐标（1280x960 4:3 实测 x 600-730 / y 835-915，留余量；
# 16:9 的 HUD 相对位置应一致但尚未实测——README 已注明）
DEFAULT_ROI = {"x": 0.455, "y": 0.862, "w": 0.135, "h": 0.098}

GROUP_SEC = 2.5      # 门控事件聚类窗口：一次击杀的 播报/奖牌/墨迹 事件簇
# 阶梯读数 offset（相对门控时刻）：负 offset 抓「弹出窗口早于门控」的杀
# （启发式墨迹检测滞后 ~0.5s），正 offset 延伸到 +2.2s 抓慢热卡
READ_LADDER = (-0.7, -0.3, 0.5, 1.2, 2.2)
MIN_CONF = 0.85      # 数字读数置信度下限（实测真读 0.9+；误读多在 0.6-0.8）
MAX_KILLS = 5        # CS2 5v5 一回合最多 5 杀：读到 6-9 必是误读，直接丢弃
STALE_SEC = 8.0      # 连续 >8s 读不到任何卡牌 = 回合已结束，旧计数作废
UPSCALE = 3          # 卡牌数字只有 ~15px，OCR 前放大 3 倍


def _read_digit(video: Path, t: float, roi: tuple[int, int, int, int],
                engine) -> tuple[int, float] | None:
    """OCR the card stack once at time t; return (count, conf) or None.

    Takes the RIGHTMOST valid digit box - the fan grows rightward, so the
    rightmost card is the newest and carries the current round count.
    """
    x, y, w, h = roi
    from .skull import _grab_frame
    img = _grab_frame(video, t, f"crop={w}:{h}:{x}:{y},scale={w*UPSCALE}:{h*UPSCALE}:flags=lanczos",
                      w * UPSCALE, h * UPSCALE)
    if img is None:
        return None
    res = engine(img)
    if res is None or res.txts is None:
        return None
    best: tuple[int, float, float] | None = None   # (cx, digit, conf)
    boxes = list(res.boxes) if res.boxes is not None else []
    scores = list(res.scores) if res.scores is not None else [1.0] * len(res.txts)
    for box, txt, score in zip(boxes, res.txts, scores):
        s = str(txt).strip()
        if len(s) != 1 or not s.isdigit():
            continue
        d = int(s)
        if not (1 <= d <= MAX_KILLS):
            continue                # 0/6-9：误读（5v5 回合最多 5 杀）
        conf = float(score)
        if conf < MIN_CONF:
            continue
        xs = [p[0] for p in box]
        cx = sum(xs) / len(xs)
        if best is None or cx > best[0]:
            best = (cx, d, conf)
    if best is None:
        return None
    return best[1], best[2]


def _group_gates(gate_events: list[dict], duration: float) -> list[float]:
    """Collapse recall events into group start times (one ladder per burst)."""
    ts = sorted(e["t"] for e in gate_events)
    groups: list[float] = []
    for t in ts:
        if groups and t - groups[-1] <= GROUP_SEC:
            continue
        groups.append(t)
    return [g for g in groups if g + READ_LADDER[-1] < duration and g + READ_LADDER[0] >= 0]


def detect_killcard(video: str | Path, cfg: dict, meta: dict,
                    gate_events: list[dict]) -> dict:
    """Track the CS2 round-kill card around recall events.

    Returns {"events", "reads", "kills"}; events carry source="killcard",
    confirmed=True, kills=delta (the recorder's kills at that moment) and
    streak=round count after those kills. Returns empty for non-csgo2 games.
    """
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    if game != "csgo2":
        return {"events": [], "reads": 0, "kills": 0}
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise KillcardNotReady(
            f"击杀卡牌通道需要 rapidocr（{exc}）。"
            "安装: pip install -r requirements-ocr.txt") from exc

    kc = cfg.get("killcard", {})
    roi_cfg = {**DEFAULT_ROI, **(kc.get("roi") or {})}
    x = int(roi_cfg["x"] * meta["width"])
    y = int(roi_cfg["y"] * meta["height"])
    w = int(roi_cfg["w"] * meta["width"])
    h = int(roi_cfg["h"] * meta["height"])
    w = min(w, meta["width"] - x)
    h = min(h, meta["height"] - y)
    roi = (x, y, w, h)

    engine = RapidOCR(params={"Global.max_side_len": 640,
                              "Global.text_score": 0.4})
    # 注意：必须在引擎创建之后再压日志级别——RapidOCR 初始化会重置 logger
    # 为 INFO 并强加 handler（v0.4.0 起的静音失效根因，实测验证）
    import logging
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    video = Path(video)
    duration = float(meta["duration"])
    gates = _group_gates(gate_events, duration)
    log(f"CS2 击杀卡牌识别：卡牌区 {w}x{h}@({x},{y})，{len(gates)} 个召回事件组"
        f"（阶梯 {len(READ_LADDER)} 点读数）", "killcard")

    events: list[dict] = []
    reads = 0
    prev: int | None = None           # 上一次确认的回合计数（回合内单调递增）
    last_card_t = -1e9                # 最近一次读到卡牌的时刻
    empty_streak = 0                  # 连续读不到卡牌的门控组数（回合结束证据）
    for g in gates:
        series: list[tuple[float, int]] = []   # (offset, count) 有效读数序列
        best_conf = 0.0
        for off in READ_LADDER:
            r = _read_digit(video, g + off, roi, engine)
            reads += 1
            if r is not None:
                series.append((off, r[0]))
                best_conf = max(best_conf, r[1])
        if not series:
            empty_streak += 1         # 本组无卡牌：可能回合已结束（也可能偶发不可读）
            continue
        # 回合结束判定：连续多组无卡牌 + 距上张卡太久 → 旧计数作废。
        # 只用"见过空"的证据，不用门控间隔（间隔大 ≠ 卡牌消失）
        if prev is not None and empty_streak >= 2 and g - last_card_t > STALE_SEC:
            prev = None
        empty_streak = 0
        last_card_t = g
        v_end = max(v for _, v in series)
        values = [v for _, v in series]
        rising = any(values[i] < values[j] for i in range(len(values))
                     for j in range(i + 1, len(values)))
        v_first = values[0]

        if prev is None:
            # 无基线：卡牌可能是视频开始前的历史状态（保守 1 杀）；
            # 但阶梯内出现递增（如 1→2→3）说明这些卡就在本窗口内弹出
            delta = (v_end - v_first + 1) if rising else 1
        elif v_end == prev:
            if rising and v_first < prev:
                # 回合重置后在本窗口内重建到与上回合相同的数字（如 3→1→2→3）
                delta = v_end - v_first + 1
            else:
                continue              # 数字没变：门控是别人的击杀/HUD 噪声
        elif v_end > prev:
            if v_end - prev > MAX_KILLS - 1:
                continue              # 跳变过大：必是误读
            if rising and v_first < prev:
                # 阶梯显示回合重置后一路涨上来（v_first < prev 而终值反而更高）
                delta = v_end - v_first + 1
            else:
                delta = v_end - prev
        else:
            # v_end < prev：回合已重置，旧卡消失、新卡从 1 重新数起
            delta = (v_end - v_first + 1) if rising else v_end
        events.append({
            "t": round(g, 3), "source": "killcard",
            "strength": 0.9, "confirmed": True,
            "kills": delta, "streak": v_end,
            "card_conf": round(best_conf, 3),
        })
        prev = v_end

    total = sum(e["kills"] for e in events)
    log(f"击杀卡牌识别完成：{reads} 次读数 → {len(events)} 组玩家击杀（共 {total} 杀）"
        + (f"，最高本回合 {max(e['streak'] for e in events)} 杀" if events else ""), "killcard")
    return {"events": events, "reads": reads, "kills": total}
