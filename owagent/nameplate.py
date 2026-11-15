"""L1 nameplate channel (Overwatch): who is the recorder?

OW2 carries the player's name tag at the BOTTOM-LEFT of the HUD - hero
portrait, level badge and the name text next to it. It is the only place
where the recorder's own name is printed continuously, which makes it a
far more direct source of "who is the target user" than inferring the
name from kill-feed votes.

It also doubles as a POV signal: while spectating (after death), the
nameplate switches to the VIEWED player's name + portrait, exactly like
the CS2 bottom info card. A stretch of stable non-owner names means the
footage is through someone else's eyes.

Pipeline position: BEFORE the OCR channel runs, so the name it reads is
injected into cfg["ocr"]["player_name"] and every downstream consumer
(kill-feed own-row judgement, player-name canonicalisation, VLM player
hints) inherits it. Together with the kill feed this becomes the user's
requested cross-check: a feed row whose KILLER name matches the
nameplate name is the recorder's own kill.

Measured on 1920x1080 footage: nameplate box spans roughly
x 60-560 / y 920-1060 (portrait + name + hp digits). OCR noise to
filter: hp pairs ("525|525"), level digits ("1"/"2" or glued to the
name "1LDWMD"), lobby/heroes UI words, single-letter combat junk.

Dependencies: rapidocr - same as the OCR channel; degrades gracefully.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .util import NO_WINDOW, ToolError, ffmpeg_bin, log, read_exact


class NameplateNotReady(ToolError):
    """The nameplate channel cannot run (missing rapidocr)."""


# 左下角名牌 ROI（相对坐标，1920x1080 实测 x60-560 / y920-1060）
PLATE_ROI = {"x": 0.031, "y": 0.852, "w": 0.26, "h": 0.13}

FPS = 0.5                # 名牌是常驻 HUD：0.5fps 足够
STILL_DIFF = 3.0         # 帧差低于此值跳过 OCR（画面没变）
MIN_READS = 5            # 主名至少出现的读数次数（防单帧噪声成主名）
NAME_SIM = 0.72          # 拼写聚类阈值（LDWMD/LDDWMD/LOWMD 归一簇）
MATCH_SIM = 0.45         # killer 名与主名的匹配下限（低于此=不是同一人）
POV_MIN_HITS = 3         # 连续同名非主名读数 ≥3 次 → 观战他人区间
POV_MAX_GAP = 4.0        # 区间内读数间隔上限（超过则断开）
POV_PAD = 2.0            # 观战区间前后外扩秒数

# 名牌区会读到的非玩家名词（大厅/英雄选择界面、状态提示）
_NOT_NAME = ("车", "聊天", "英雄", "信息", "铁壁", "威能", "选择", "管理",
             "账号", "退回", "桌面", "竞技", "休闲", "比赛", "生涯", "生涯",
             "战绩", "组队", "好友", "左ALT", "ALT", "F1", "MOUSE", "CTRL",
             "观战", "视角", "回放", "最佳", "精彩")

_LEAD_NUM = re.compile(r"^[\d\s│|.:·+]+")


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _is_name(text: str) -> bool:
    """A plausible player-name token from the nameplate OCR."""
    t = str(text).strip()
    if not (3 <= len(t) <= 24):
        return False
    if "|" in t or "→" in t or "➤" in t:
        return False                      # 血量对 / 播报箭头
    if any(w in t.upper() for w in ("ALT", "F1", "MOUSE")):
        return False
    if any(w in t for w in _NOT_NAME):
        return False
    alnum = sum(1 for ch in t if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    if alnum / len(t) < 0.8:
        return False                      # "zs4+" 类战斗噪声（真实名几乎纯字母数字）
    digits = sum(1 for ch in t if ch.isdigit())
    if digits / len(t) > 0.4:
        return False                      # 数字为主：等级/血量
    return any("\u4e00" <= ch <= "\u9fff" or ch.isalpha() for ch in t)


def _extract_name(token: str) -> str:
    """Strip the glued level digit ("1LDWMD" / "1 LDWMD") → "LDWMD"."""
    t = _LEAD_NUM.sub("", str(token).strip())
    return t if _is_name(t) else ""


def _frame_names(result) -> list[str]:
    """Name-like tokens in one OCR result (usually 0 or 1)."""
    if result is None or not getattr(result, "txts", None):
        return []
    out: list[str] = []
    for txt in result.txts:
        n = _extract_name(txt)
        if n:
            out.append(n)
    return out


def _vote_player(reads: list[tuple[float, str]]) -> tuple[str, list[str]]:
    """Fuzzy-cluster name reads; return (dominant spelling, cluster names)."""
    if not reads:
        return "", []
    clusters: list[tuple[str, list[tuple[float, str]]]] = []   # (rep, members)
    for t, n in reads:
        nn = _norm(n)
        for i, (rep, members) in enumerate(clusters):
            if _similar(nn, _norm(rep)) >= NAME_SIM:
                members.append((t, n))
                break
        else:
            clusters.append((n, [(t, n)]))
    # 主簇：读数次数最多；并列时取时间跨度更长的
    def _key(c):
        ts = [t for t, _ in c[1]]
        return (len(c[1]), max(ts) - min(ts))
    best = max(clusters, key=_key)
    rep, members = best
    counts: dict[str, int] = {}
    for _, n in members:
        counts[n] = counts.get(n, 0) + 1
    player = max(counts, key=counts.get)
    # 其余簇的全部读数（供他人视角检测用）
    others = [(t, n) for c in clusters for (t, n) in c[1] if c is not best]
    return player, others


def _other_pov_stretches(others: list[tuple[float, str]], player: str,
                         duration: float) -> list[tuple[float, float]]:
    """Continuous stretches showing the same non-owner name → spectate POV."""
    if not others or not player:
        return []
    # 按"展示的名字"聚簇（同一被观战者的拼写变体归一）
    stretches: list[list[tuple[float, str]]] = []
    for t, n in sorted(others):
        if stretches and _similar(_norm(n), _norm(stretches[-1][-1][1])) >= NAME_SIM \
                and t - stretches[-1][-1][0] <= POV_MAX_GAP:
            stretches[-1].append((t, n))
        else:
            stretches.append([(t, n)])
    out: list[tuple[float, float]] = []
    for s in stretches:
        if len(s) < POV_MIN_HITS:
            continue
        lo, hi = s[0][0], s[-1][0]
        out.append((max(0.0, lo - POV_PAD), min(duration, hi + POV_PAD)))
    # 合并相邻
    merged: list[list[float]] = []
    for lo, hi in out:
        if merged and lo <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def detect_nameplate(video: str | Path, cfg: dict, meta: dict) -> dict:
    """Read the bottom-left nameplate: target user name + POV stretches.

    Returns {"player", "reads", "other_pov", "frames", "ocr_calls"}.
    `player` is '' when no stable name was found (degrade gracefully).
    """
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    if game != "overwatch":
        return {"player": "", "reads": [], "other_pov": [],
                "frames": 0, "ocr_calls": 0}
    try:
        from rapidocr import RapidOCR
        import numpy as np
    except ImportError as exc:
        raise NameplateNotReady(
            f"名牌通道需要 rapidocr（{exc}）。"
            "安装: pip install -r requirements-ocr.txt") from exc

    from .util import even
    npc = cfg.get("nameplate", {})
    roi = {**PLATE_ROI, **(npc.get("roi") or {})}
    x = even(int(roi["x"] * meta["width"]))
    y = even(int(roi["y"] * meta["height"]))
    w = even(min(int(roi["w"] * meta["width"]), meta["width"] - x))
    h = even(min(int(roi["h"] * meta["height"]), meta["height"] - y))
    fps = float(npc.get("fps", FPS))
    still = float(npc.get("still_diff", STILL_DIFF))

    engine = RapidOCR(params={"Global.max_side_len": 480,
                              "Global.text_score": 0.4})
    # RapidOCR 初始化会重置 logger：必须在引擎创建之后压级别（v0.7.1 教训）
    import logging
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)

    log(f"OW 名牌识别：左下角 ROI x={x} y={y} w={w} h={h} @ {fps}fps", "nameplate")
    frame_size = w * h * 3
    vf = f"fps={fps},crop={w}:{h}:{x}:{y}"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    reads: list[tuple[float, str]] = []
    idx = 0
    ocr_calls = 0
    prev = None
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
                img = np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3)
                diff = (float(np.abs(img.astype(np.int16)
                                     - prev.astype(np.int16)).mean())
                        if prev is not None else 1e9)
                prev = img
                if diff < still:
                    continue               # 画面没变（名牌常驻）
                ocr_calls += 1
                for n in _frame_names(engine(img)):
                    reads.append((t, n))
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-800:]
            log(f"名牌通道解码失败: {detail}", "warn")

    player, others = _vote_player(reads)
    # 主名稳定性：读数次数不足时不采信（大厅残影/战斗噪声）
    player_reads = sum(1 for _, n in reads if _similar(_norm(n), _norm(player)) >= NAME_SIM)
    if player and player_reads < MIN_READS:
        log(f"名牌通道：候选名「{player}」仅 {player_reads} 次读数（< {MIN_READS}），不采信",
            "nameplate")
        player = ""
    other_pov = _other_pov_stretches(others, player, float(meta["duration"])) if player else []

    if player:
        span = [t for t, n in reads if _similar(_norm(n), _norm(player)) >= NAME_SIM]
        log(f"名牌通道：目标用户「{player}」（{player_reads} 次读数，"
            f"{min(span):.0f}s~{max(span):.0f}s）", "nameplate")
        if other_pov:
            log(f"名牌通道：检测到 {len(other_pov)} 段他人视角区间"
                f"（{' / '.join(f'{lo:.0f}-{hi:.0f}s' for lo, hi in other_pov)}）", "nameplate")
    else:
        log("名牌通道：未读到稳定玩家名（保持原玩家推断逻辑）", "nameplate")

    return {"player": player, "reads": [{"t": round(t, 1), "name": n} for t, n in reads],
            "other_pov": other_pov, "frames": idx, "ocr_calls": ocr_calls}


def confirm_feed_rows(events: list[dict], player: str) -> int:
    """Upgrade kill-feed rows whose killer IS the target user to confirmed.

    The feed prints EVERYONE's kills, but a row whose killer name matches
    the nameplate name is by definition the recorder's own kill. Used in
    full OCR mode so name-matched rows join the timeline as confirmed
    events. Returns the number of upgraded events.
    """
    upgraded = 0
    for e in events:
        if e.get("confirmed") or e.get("source") != "ocr":
            continue
        killer = str(e.get("killer") or "").strip()
        if killer and _similar(_norm(killer), _norm(player)) >= MATCH_SIM:
            e["confirmed"] = True
            e["player_kill"] = True
            upgraded += 1
    return upgraded


def cross_check_killer(events: list[dict], player: str) -> tuple[int, int]:
    """Nameplate x kill-feed cross-check on confirmed skull events.

    A confirmed skull event carries the killer name read from the feed
    row that appeared right after the skull. When the nameplate channel
    knows the target user, that killer name should match it:
      * match (>= MATCH_SIM)  -> keep confirmed (two independent signals agree)
      * mismatch              -> demote (shape false positive / misread feed)
    Returns (kept, demoted). Events with no killer name are left alone.
    """
    kept = demoted = 0
    for e in events:
        if not e.get("confirmed"):
            continue
        killer = str(e.get("killer") or "").strip()
        if not killer or not player:
            continue
        if _similar(_norm(killer), _norm(player)) >= MATCH_SIM:
            kept += 1
        else:
            e["confirmed"] = False
            e["name_mismatch"] = killer
            demoted += 1
    return kept, demoted
