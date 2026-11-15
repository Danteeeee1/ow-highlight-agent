"""L1 OCR channel: read the kill feed text and turn it into named kill events.

The heuristic (ink statistics) and Crispy (kill-award medal) channels both
answer "something happened around t". Neither can answer "who killed whom,
and how many in a row" - that is exactly what the VLM kept mis-counting.

This module decodes the kill-feed ROI at a low fps (default 1) and runs
RapidOCR (PP-OCRv6, fully offline, onnxruntime CPU) on it. Two cost gates
keep it practical (CPU OCR is ~1.5s/frame):

  * event windows - OCR only runs around timestamps where the other recall
    channels saw feed activity (the heuristic ink-rise watches the same ROI,
    so a text change without an ink rise is nearly impossible);
  * frame diff - inside a window, unchanged frames still skip OCR.

Each kill-feed row is parsed into (killer, victim); rows are de-duplicated
across frames with fuzzy matching (OCR wobbles: "DAnTe" vs "DAnTo"), so one
on-screen row = one kill event like::

    {"t": 232.0, "source": "ocr", "strength": 0.9,
     "killer": "DAnTe", "victim": "暖暖不吃香菜", "streak": 3}

Downstream value:
  * deterministic multi-kill counting (same killer within `streak_window`),
  * real player names for VLM hints / captions,
  * independent recall channel that catches kills the other two miss.

Dependencies (all optional, pipeline degrades gracefully without them):
rapidocr + onnxruntime + opencv - see requirements-ocr.txt.
"""
from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from pathlib import Path

from .config import pixel_roi
from .util import NO_WINDOW, ToolError, ffmpeg_bin, log, read_exact


class OCRNotReady(ToolError):
    """The OCR channel cannot run (missing rapidocr / deps)."""


# ROI 顶部跳过比例：CS2 ROI 已下移避开网络统计条，留少量余量即可
FEED_TOP_SKIP = {"csgo2": 0.08, "overwatch": 0.0, "valorant": 0.05}

# 纯数字/符号/性能指标的文本不是玩家名（FPS、延迟、丢包率等 HUD 噪声）
_NOISE_WORDS = re.compile(r"延迟|丢包|最高|平场|平均|fps|ping|loss|choke|ms", re.I)

KILL_FEED_TEXT_MIN = 2      # 玩家名至少 2 个字符
KILL_FEED_TEXT_MAX = 24     # 超过 24 字符的基本是播报横幅而非玩家名
FUZZY_SAME = 0.62           # 归一化文本相似度超过该值视为同一段文字
GATE_BEFORE = 1.0           # 门控事件前扩几秒
GATE_AFTER = 10.0           # 门控事件后扩几秒（播报在屏停留时间）


def _digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    digits = sum(1 for ch in text if ch.isdigit() or ch in ".%|:/_- ")
    return digits / len(text)


def _is_plausible_name(text: str) -> bool:
    """Filter kill-feed candidates from HUD noise (FPS counters, latency)."""
    t = text.strip()
    if not (KILL_FEED_TEXT_MIN <= len(t) <= KILL_FEED_TEXT_MAX):
        return False
    if _NOISE_WORDS.search(t):
        return False
    if _digit_ratio(t) > 0.6:
        return False
    return any("\u4e00" <= ch <= "\u9fff" or ch.isalpha() for ch in t)


def _norm_name(text: str) -> str:
    """Lowercase alnum+CJK only - fuzzy compare keys."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _parse_rows(result, top_skip_px: int) -> list[tuple[str, str, float, int, int]]:
    """Group OCR boxes into lines; a 2+ name line is (killer, victim).

    A real kill-feed row has a wide gap between killer and victim (the weapon
    icon sits there). A row fading out often OCR-splits ONE name into two
    adjacent boxes - the gap check rejects those.
    """
    if result is None or not getattr(result, "txts", None):
        return []
    boxes = list(result.boxes) if result.boxes is not None else []
    texts = list(result.txts)
    scores = list(result.scores) if result.scores is not None else [1.0] * len(texts)
    items = []
    for box, text, score in zip(boxes, texts, scores):
        if not _is_plausible_name(str(text)):
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        cy = sum(ys) / len(ys)
        if cy < top_skip_px:
            continue  # 网络统计条等顶部噪声
        items.append({"cx": sum(xs) / len(xs), "cy": cy,
                      "x0": min(xs), "x1": max(xs),
                      "h": max(ys) - min(ys), "text": str(text).strip(),
                      "score": float(score)})
    if not items:
        return []
    items.sort(key=lambda a: a["cy"])
    lines: list[list[dict]] = []
    for it in items:
        if lines and abs(it["cy"] - lines[-1][0]["cy"]) <= max(it["h"], 14.0):
            lines[-1].append(it)
        else:
            lines.append([it])
    rows = []
    for line in lines:
        if len(line) < 2:
            continue  # 单个文本框：无法构成“击杀者→受害者”，多为噪声
        line.sort(key=lambda a: a["cx"])
        killer, victim = line[0], line[-1]
        gap = victim["x0"] - killer["x1"]
        if gap < 10.0:
            continue  # 两个文本框紧贴：大概率是一个名字被拦腰截断
        if _norm_name(killer["text"]) == _norm_name(victim["text"]):
            continue
        rows.append((killer["text"], victim["text"],
                     min(killer["score"], victim["score"])))
    return rows


def _match_active(killer: str, victim: str, active: dict):
    """Find the active row this OCR result most likely belongs to."""
    nk, nv = _norm_name(killer), _norm_name(victim)
    best, best_score = None, 0.0
    for key, info in active.items():
        s = min(_similar(nk, key[0]), _similar(nv, key[1]))
        if s > best_score:
            best, best_score = key, s
    return best if best_score >= FUZZY_SAME else None


def _is_fade_artifact(killer: str, victim: str, active: dict) -> bool:
    """A row fading out often OCR-splits one long name into a bogus
    killer→victim pair. The tell: BOTH sides of the new row resemble the
    SAME on-screen name (e.g. "此时若是君在便" + "何须望" both come from
    "此时若是君在侧，何须望..."). A real kill has two different entities."""
    nk, nv = _norm_name(killer), _norm_name(victim)
    for key in active:
        for name in key:
            if len(name) < 4 or name in (nk, nv):
                continue
            k_hit = (_similar(nk, name) >= 0.6
                     or (2 <= len(nk) < len(name) and nk in name))
            v_hit = (_similar(nv, name) >= 0.6
                     or (2 <= len(nv) < len(name) and nv in name))
            if k_hit and v_hit:
                return True
    return False


def _canon_killers(kills: list[dict], threshold: float = 0.8) -> None:
    """Fold OCR spelling wobble ("DAnTe"/"DAnTo") into one killer key.

    Mutates each kill dict with a "killer_key" used for streak counting.
    """
    reps: list[str] = []
    for k in kills:
        nk = _norm_name(k["killer"])
        rep = next((r for r in reps if _similar(nk, _norm_name(r)) >= threshold), None)
        if rep is None:
            reps.append(k["killer"])
            k["killer_key"] = k["killer"]
        else:
            k["killer_key"] = rep


def _infer_player(kills: list[dict], model_events: list[dict],
                  window: float = 2.5) -> str:
    """The recorder's kills are the ones the Crispy medal channel confirms."""
    if not kills:
        return ""
    counts: dict[str, int] = {}
    for k in kills:
        if any(abs(k["t"] - m["t"]) <= window for m in model_events):
            counts[k["killer_key"]] = counts.get(k["killer_key"], 0) + 1
    if counts:
        return max(counts, key=counts.get)
    return ""


def _gate_windows(gate_events: list[dict], duration: float) -> list[tuple[float, float]]:
    """Merge gate events into [start, end] OCR sampling windows."""
    spans = sorted((max(0.0, e["t"] - GATE_BEFORE),
                    min(duration, e["t"] + GATE_AFTER)) for e in gate_events)
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _in_windows(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(lo <= t <= hi for lo, hi in windows)


def detect_ocr(video: str | Path, cfg: dict, meta: dict,
               model_events: list[dict] | None = None,
               gate_events: list[dict] | None = None,
               windows: list[tuple[float, float]] | None = None) -> dict:
    """Read the kill feed inside `windows` (or gate events / whole video).

    Returns {"events", "kills", "player", ...}; every kill event carries
    killer / victim / streak, deterministic data the other channels lack.
    """
    oc = cfg.get("ocr", {})
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    try:
        from rapidocr import RapidOCR
        import numpy as np
    except ImportError as exc:
        raise OCRNotReady(
            f"OCR 通道需要 rapidocr（{exc}）。"
            "安装: pip install -r requirements-ocr.txt") from exc

    x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
    fps = float(oc.get("fps", 1.0))
    still_diff = float(oc.get("still_diff", 2.5))
    min_score = float(oc.get("min_score", 0.5))
    streak_window = float(oc.get("streak_window", 6.0))
    persist_sec = float(oc.get("feed_persist_sec", 25.0))
    player = str(oc.get("player_name") or "").strip()
    top_skip_px = int(h * FEED_TOP_SKIP.get(game, 0.0))

    # 采样窗口优先级：显式 windows > 门控事件窗口 > 全片
    if windows is None:
        windows = (_gate_windows(gate_events or [], meta["duration"])
                   if gate_events else [])
    if not windows:
        mode = "全片扫描（慢，无窗口信息）"
    else:
        mode = f"{len(windows)} 个窗口共 {sum(hi - lo for lo, hi in windows):.0f}s"

    engine = RapidOCR(params={"Global.max_side_len": int(oc.get("max_side_len", 320)),
                              "Global.text_score": min_score})
    # 必须在引擎创建之后再压日志级别：RapidOCR 初始化会把 logger 重置为 INFO
    # 并强加自己的 StreamHandler——"先设级别再建引擎"的旧写法会被覆盖（静音失效根因）
    import logging
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    frame_size = w * h * 3
    vf = f"fps={fps},crop={w}:{h}:{x}:{y}"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    log(f"OCR 击杀播报识别（{game}）：ROI x={x} y={y} w={w} h={h} @ {fps}fps，{mode}", "ocr")

    # (norm_killer, norm_victim) -> first-seen t；OCR 抖动用模糊匹配归并
    active: dict[tuple[str, str], dict] = {}
    kills: list[dict] = []
    signals: list[dict] = []
    prev = None
    idx = 0
    ocr_calls = 0

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
                idx += 1
                if windows and not _in_windows(t, windows):
                    continue  # 窗口之外：这段不可能有播报变化
                img = np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3)
                diff = (float(np.abs(img.astype(np.int16)
                                     - prev.astype(np.int16)).mean())
                        if prev is not None else 1e9)
                prev = img
                signals.append({"t": round(t, 3), "feed_diff": round(diff, 2)})
                if diff < still_diff:
                    continue  # 播报区没变：跳过 OCR
                ocr_calls += 1
                rows = _parse_rows(engine(img), top_skip_px)
                seen_now: list[tuple[str, str]] = []
                for killer, victim, score in rows:
                    key = _match_active(killer, victim, active)
                    dup_in_frame = key in seen_now if key else False
                    fresh = (key is None
                             or t - active[key]["first"] > persist_sec
                             or dup_in_frame)
                    if key is None and _is_fade_artifact(killer, victim, active):
                        continue  # 淡出行被拦腰截断产生的假“击杀”
                    if fresh:
                        kills.append({"t": round(t, 3), "killer": killer,
                                      "victim": victim, "conf": round(score, 3)})
                        active[(_norm_name(killer), _norm_name(victim))] = {
                            "first": t, "last": t}
                        seen_now.append((_norm_name(killer), _norm_name(victim)))
                    elif key is not None:
                        active[key]["last"] = t
                        seen_now.append(key)
                # 滚出屏幕的行：过期回收
                for key in list(active):
                    if active[key]["last"] < t - persist_sec:
                        del active[key]
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-1200:]
            log(f"OCR 解码失败: {detail}", "warn")

    _canon_killers(kills)
    if not player:
        player = _infer_player(kills, model_events or [])

    # streak: same killer_key within streak_window before (and including) this kill
    events: list[dict] = []
    for i, k in enumerate(sorted(kills, key=lambda a: a["t"])):
        streak = 1 + sum(1 for j in range(i)
                         if kills[j]["killer_key"] == k["killer_key"]
                         and k["t"] - kills[j]["t"] <= streak_window)
        k["streak"] = streak
        own = (not player) or k["killer_key"] == player
        events.append({
            "t": k["t"], "source": "ocr",
            "strength": round(0.55 + 0.35 * own + 0.05 * min(streak, 3), 3),
            "killer": k["killer"], "victim": k["victim"],
            "streak": streak, "ocr_conf": k["conf"],
        })

    top = max(((k["streak"], k["killer_key"]) for k in kills), default=(0, ""))
    log(f"OCR 扫描 {idx} 帧（{idx / fps:.0f}s，实际 OCR {ocr_calls} 次），"
        f"读到 {len(kills)} 条击杀播报"
        + (f"，玩家「{player}」最高连杀 {top[0]}" if player and top[0] > 1 else ""), "ocr")
    return {"events": events, "signals": signals, "frames": idx, "fps": fps,
            "kills": kills, "player": player, "backend": "rapidocr",
            "ocr_calls": ocr_calls, "roi_px": [x, y, w, h]}


def annotate_selected(video: str | Path, cfg: dict, meta: dict, tl: dict,
                      model_events: list[dict] | None = None) -> dict:
    """Annotate mode: OCR only the SELECTED segments of a built timeline.

    Multi-kill judgement only matters inside candidate clips, and recall is
    already covered by the heuristic + Crispy channels (same ROI), so the
    default mode reads the feed only where it will be used - a 5-minute
    video costs ~1-3 minutes instead of a full scan.
    """
    segs = [s for s in tl.get("selected", []) if s.get("duration", 0) > 0]
    if not segs:
        return {"events": [], "kills": [], "player": "", "segments": 0}
    windows = [(float(s["start"]), float(s["end"])) for s in segs]
    res = detect_ocr(video, cfg, meta, model_events=model_events, windows=windows)

    kills = res["kills"]
    player = res["player"]
    for seg in segs:
        lo, hi = float(seg["start"]) - 1.0, float(seg["end"])
        ks = [{"t": k["t"], "killer": k["killer"], "victim": k["victim"],
               "streak": k["streak"]} for k in kills if lo <= k["t"] <= hi]
        seg["ocr_kills"] = ks
        max_streak = max((k["streak"] for k in ks), default=0)
        if ks:
            streak_killers = {k["killer"] for k in ks if k["streak"] == max_streak}
            lead = player if player in streak_killers else sorted(streak_killers)[0]
            bit = f"OCR 读到 {len(ks)} 条击杀（{lead}）"
            if max_streak >= 2:
                bit += f"，最高 {max_streak} 连杀"
            seg["reason"] = f"{bit}、{seg['reason']}"
            seg["caption"] = f"HIGHLIGHT {seg.get('rank', 1):02d}  {bit}"
    res["segments"] = len(segs)
    return res
