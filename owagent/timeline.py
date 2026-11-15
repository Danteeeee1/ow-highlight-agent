"""L2/L3: merge recalled events into candidate clips, score, select Top-K."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

# 三个"看见击杀"的召回源（区别于音频峰值）：聚类、波次拆分、窗口切分都按它们算
VIS_SOURCES = ("visual", "model", "ocr", "skull", "killcard")


def _split_waves(events: list[dict], wave_gap: float) -> list[list[dict]]:
    """Split one cluster into "waves" - bursts of kills separated by calm.

    Audio peaks (gunfire/footsteps) chain continuously in shooters, so a
    90-second cluster is common even when it contains several distinct fight
    waves. Kills (visual/model/ocr events) are sparse in comparison: a gap of
    `wave_gap` seconds without any kill marks a wave boundary. Each wave then
    becomes its own candidate, so one giant cluster can no longer swallow a
    mid-cluster triple kill by trimming everything to a single max_len clip.
    """
    vis_ts = [e["t"] for e in events if e.get("source") in VIS_SOURCES]
    if len(vis_ts) < 2:
        return [events]
    cuts = []
    for a, b in zip(vis_ts, vis_ts[1:]):
        if b - a > wave_gap:
            cuts.append((a + b) / 2.0)
    if not cuts:
        return [events]
    waves: list[list[dict]] = []
    cur: list[dict] = []
    for e in events:  # events are time-sorted
        if cuts and e["t"] > cuts[0]:
            waves.append(cur)
            cur = [e]
            cuts.pop(0)
        else:
            cur.append(e)
    waves.append(cur)
    return [w for w in waves if w]


def _subtract(start: float, end: float,
              intervals: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    """Clip [start, end] against exclude intervals; return surviving pieces.

    Used to drop spectator-view stretches (someone else's POV) from a
    candidate window before it is emitted.
    """
    if not intervals:
        return [(start, end)]
    pieces: list[tuple[float, float]] = []
    cur = start
    for lo, hi in sorted(intervals):
        if hi <= cur or lo >= end:
            continue
        if lo > cur:
            pieces.append((cur, min(lo, end)))
        cur = max(cur, hi)
    if cur < end:
        pieces.append((cur, end))
    return [(s, e) for s, e in pieces if e - s >= 0.5]


def build_timeline(video: str | Path, visual_events: list[dict], audio_events: list[dict],
                   duration: float, cfg: dict, ocr: dict | None = None,
                   exclude_intervals: list[tuple[float, float]] | None = None,
                   sources: list[dict] | None = None) -> dict:
    """Build the highlight timeline.

    Single-source mode: events carry plain local times (``src`` defaults to 0).
    Multi-source mode: events are pre-offset onto a virtual concatenation
    timeline (``e["t"] = offset_i + local_t``) and tagged ``e["src"] = i``;
    ``sources`` lists ``{"offset", "duration", "video"}`` per input. Clusters
    never span a source boundary, and every candidate keeps LOCAL times in
    ``start/end/event_times/...`` plus ``src``/``src_video`` so rendering,
    VLM review and OCR annotation can go back to the right file.
    """
    clip = cfg["clip"]
    sel = cfg["select"]
    pre, post = float(clip["pre_roll"]), float(clip["post_roll"])
    gap, min_len, max_len = float(clip["merge_gap"]), float(clip["min_len"]), float(clip["max_len"])

    if not sources:
        sources = [{"offset": 0.0, "duration": duration, "video": str(video)}]

    def _src_of(e: dict) -> int:
        return int(e.get("src", 0) or 0)

    def _bounds(src: int) -> tuple[float, float]:
        """(offset, end) of that source on the virtual timeline."""
        if 0 <= src < len(sources):
            s = sources[src]
            return float(s["offset"]), float(s["offset"]) + float(s["duration"])
        return 0.0, duration

    all_events = sorted(visual_events + audio_events, key=lambda e: e["t"])
    clusters: list[dict] = []
    for e in all_events:
        # 跨素材强制断开：不同视频的事件绝不聚成一段（虚拟时间轴相邻
        # 素材的首尾事件间隔可能小于 merge_gap，必须显式按 src 切）
        if (clusters and e["t"] - clusters[-1]["last"] <= gap
                and _src_of(e) == clusters[-1]["src"]):
            c = clusters[-1]
            c["events"].append(e)
            c["last"] = e["t"]
        else:
            clusters.append({"events": [e], "first": e["t"], "last": e["t"],
                             "src": _src_of(e)})

    # 玩家中心性：击杀卡牌/骷髅头是 HUD 私有元素（只有录像主人本人能看到），
    # 一旦确认过"玩家本人的击杀"，成片就该以玩家为主角——无玩家击杀的段
    # （别人的连杀/纯架枪/换弹）降权。整段录像都没确认过玩家击杀时（通道
    # 未启用/玩家零杀）不降权，避免把纯音频精彩段误杀。
    has_player_kills = any(e.get("confirmed") for e in visual_events)
    playerless_penalty = float(sel.get("playerless_penalty", 0.15))

    # 长簇按击杀波次拆分：音频峰值在射击游戏里几乎连续，会把多场战斗
    # 连成一个几十秒的大簇；按视觉/模型事件的间隙切开，一波战斗一个候选。
    wave_gap = gap + 3.0
    wave_lists = [_split_waves(c["events"], wave_gap) for c in clusters]
    waves = [w for ws in wave_lists for w in ws]

    candidates = []

    def _emit(events: list[dict], start: float, end: float) -> None:
        vis = [e for e in events if e["source"] in VIS_SOURCES]
        aud = [e for e in events if e["source"] == "audio"]
        mdl = [e for e in events if e["source"] == "model"]
        src = _src_of(events[0]) if events else 0
        off, src_end = _bounds(src)
        # 骷髅头/击杀卡牌确认 = 录像玩家的击杀（这些 HUD 只对击杀者本人显示），
        # 是比音频佐证更强的多通道证据；killcard 事件带 kills 增量（一跳多杀）
        confirmed = [e for e in events if e.get("confirmed")]
        confirmed_kills = sum(e.get("kills", 1) for e in confirmed)
        base = max((e["strength"] for e in events), default=0.0)
        combo = min(len(vis) / 4.0, 1.0) if vis else min(len(aud) / 3.0, 1.0) * 0.5
        cross = 1.0 if (vis and aud) else 0.0
        # 片头/片尾裕度按"该素材"的边界算（多源时 start/end 是全局时间）
        margin = 1.0 if (start >= off + 5.0 and end <= src_end - 5.0) else 0.6
        # score = 0.40*peak + 0.30*combo + 0.20*cross-modal + 0.10*context
        #         (+0.10 玩家击杀确认，每多一杀 +0.05，封顶 0.20)
        kill_bonus = min(0.10 + 0.05 * max(confirmed_kills - 1, 0), 0.20) if confirmed_kills else 0.0
        score = (0.40 * base + 0.30 * combo + 0.20 * cross + 0.10 * margin
                 + kill_bonus)
        # 玩家中心性降权：有玩家击杀铁证、但这段一个都没有
        penalized = has_player_kills and not confirmed_kills and playerless_penalty > 0
        if penalized:
            score -= playerless_penalty

        # OCR 读到的具名击杀（含连杀数），供 VLM 复核提示与 L3 融合使用
        # （时间统一为素材内局部时间）
        ocr_kills = [{"t": round(e["t"] - off, 3), "killer": e.get("killer", ""),
                      "victim": e.get("victim", ""), "streak": e.get("streak", 1)}
                     for e in events if e.get("killer")]
        max_streak = max((k["streak"] for k in ocr_kills), default=0)
        # 击杀卡牌确认的玩家击杀（增量与本回合累计数），供 VLM 复核使用
        card_kills = [{"t": round(e["t"] - off, 3), "kills": e.get("kills", 1),
                       "streak": e.get("streak", 1)}
                      for e in events if "killcard" in (e.get("channels") or [])]

        bits = []
        card_events = [e for e in events
                       if "killcard" in (e.get("channels") or []) and e.get("confirmed")]
        spectated = [e for e in events if e.get("spectate")]
        if card_events:
            top = max(e.get("streak", 1) for e in card_events)
            bits.append(f"击杀卡牌确认玩家击杀（本回合累计 {top} 杀）")
        elif confirmed:
            killer = confirmed[0].get("killer") or "玩家"
            bits.append(f"骷髅头确认玩家击杀×{len(confirmed)}（{killer}）")
        if spectated:
            bits.append(f"含 {len(spectated)} 组观战视角击杀（已撤销确认）")
        if penalized:
            bits.append("无玩家击杀确认（降权）")
        if ocr_kills:
            names = sorted({k["killer"] for k in ocr_kills})
            lead = names[0] if len(names) == 1 else f"{names[0]} 等{len(names)}人"
            bits.append(f"OCR 读到 {len(ocr_kills)} 条击杀（{lead}）"
                        + (f"，最高 {max_streak} 连杀" if max_streak >= 2 else ""))
        if mdl:
            bits.append(f"{len(mdl)} 次模型判定击杀"
                        + (f"（最高 p={max(e['strength'] for e in mdl):.2f}）" if mdl else ""))
        elif vis:
            bits.append(f"{len(vis)} 次播报")
        if aud:
            bits.append(f"{len(aud)} 次音频峰值")
        if len(vis) >= 3 or max_streak >= 2:
            bits.append("疑似连杀")
        if cross:
            bits.append("视听双通道确认")
        reason = "、".join(bits) + f"，峰值强度 {base:.2f}"

        idx = len(candidates) + 1
        src_video = sources[src]["video"] if 0 <= src < len(sources) else str(video)
        candidates.append({
            "id": idx,
            # start/end 与全部时间字段都是"素材内局部时间"（渲染/复核/OCR 直接用）
            "src": src,
            "src_video": src_video,
            "start": round(start - off, 3), "end": round(end - off, 3),
            # 全局时间（虚拟拼接轴）：跨素材排序与重叠合并专用
            "g_start": round(start, 3), "g_end": round(end, 3),
            "duration": round(end - start, 3),
            "event_times": [round(e["t"] - off, 3) for e in events],
            "kill_times": [round(e["t"] - off, 3) for e in vis],
            "visual_count": len(vis), "audio_count": len(aud),
            "confirmed_kills": confirmed_kills,
            "confirmed_detail": [{"t": round(e["t"] - off, 3), "kills": int(e.get("kills", 1))}
                                 for e in confirmed],
            "ocr_kills": ocr_kills,
            "card_kills": card_kills,
            "peak_strength": round(base, 4),
            "score": round(score, 4),
            "reason": reason,
            "caption": f"HIGHLIGHT {idx:02d}  " + (" / ".join(bits[:2]) or "候选高光"),
            "selected": False,
        })

    def _emit_window(events: list[dict], start: float, end: float) -> None:
        """Emit one candidate window (splitting over-long waves as before)."""
        if end - start <= max_len:
            _emit(events, start, end)
            return
        # 超长波：按视觉事件密度产出多个 max_len 窗口（消耗式切分），
        # 每个窗口独立打分为候选，避免一波长战斗只保留 16 秒。
        # 上限 5：一波 76s 的战斗可能有 3~4 个密集击杀区（前中后段），
        # 太少会让中后段的三杀窗口永远轮不上。
        vis_ts = sorted(e["t"] for e in events if e["source"] in VIS_SOURCES)
        remaining = list(vis_ts)
        windows: list[tuple[float, float]] = []
        while remaining and len(windows) < 5:
            lo, hi = start, end - max_len
            cands = {hi} | {min(max(t - (max_len - post), lo), hi) for t in remaining}
            best = max(cands, key=lambda s: (sum(1 for t in remaining if s <= t <= s + max_len), s))
            windows.append((best, min(best + max_len, end)))
            remaining = [t for t in remaining if not (best <= t <= best + max_len)]
        # 兜底：贪心 5 窗口耗尽后仍可能有零散事件没被任何窗口覆盖
        # （实测：76s 大波 25 事件，第 6、7 个事件 187.75/189.4 被静默丢弃，
        # 导致该击杀时刻在所有候选段里消失）。每个漏网事件补一个专属窗口，
        # 宁可多一个候选也不能丢事件——召回是一切的上游。
        for t in remaining:
            lo, hi = start, end - max_len
            ws = min(max(t - (max_len - post), lo), hi)
            win = (ws, min(ws + max_len, end))
            if win not in windows:
                windows.append(win)
        if not windows:  # 无视觉事件：保留尾部（音频高潮通常在末尾）
            windows = [(end - max_len, end)]
        for ws, we in windows:
            if we - ws < 0.5:
                continue
            _emit([e for e in events if ws <= e["t"] <= we], ws, we)

    for wv in waves:
        src = _src_of(wv[0]) if wv else 0
        off, src_end = _bounds(src)
        first = min(e["t"] for e in wv)
        last = max(e["t"] for e in wv)
        start = max(off, first - pre)
        end = min(src_end, last + post)
        if end - start < min_len:
            end = min(src_end, start + min_len)

        # 观战他人视角剔除：把窗口按排除区间切开，只保留本人视角的部分；
        # 事件同样只留切后窗口内的（被观战者的击杀不进任何候选段）
        for s2, e2 in _subtract(start, end, exclude_intervals):
            if e2 - s2 < 0.5:
                continue
            wv2 = [e for e in wv if s2 <= e["t"] <= e2]
            if not wv2:
                continue
            _emit_window(wv2, s2, e2)

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    chosen: list[dict] = []
    total = 0.0
    for c in ranked:
        if len(chosen) >= int(sel["top_k"]):
            break
        if c["score"] < float(sel["min_score"]):
            continue
        if total + c["duration"] > float(sel["max_total_sec"]):
            continue
        chosen.append(c)
        total += c["duration"]
    # 成片顺序 = 素材顺序（虚拟时间轴），段上的 start/end 是素材内局部时间
    chosen.sort(key=lambda c: c.get("g_start", c["start"]))
    chosen_ids = {c["id"] for c in chosen}
    for c in candidates:
        c["selected"] = c["id"] in chosen_ids

    # Overlapping clips would repeat the same frames in the final reel.
    merged: list[dict] = []
    for c in chosen:
        item = dict(c)
        prev = merged[-1] if merged else None
        same_src = prev is not None and prev.get("src", 0) == item.get("src", 0)
        if (prev is not None and same_src
                and item.get("g_start", item["start"]) < prev.get("g_end", prev["end"])):
            off, _ = _bounds(int(prev.get("src", 0)))
            m = prev
            overlap = m.get("g_end", m["end"]) - item.get("g_start", item["start"])
            if overlap <= 2.0 and (item.get("g_end", item["end"]) - m.get("g_end", m["end"])) >= min_len * 0.5:
                # Small overlap: trim the later clip, keep it as a separate shot.
                mge = m.get("g_end", m["end"])
                item["g_start"] = mge
                item["start"] = round(mge - off, 3)
                item["duration"] = round(item.get("g_end", item["end"]) - mge, 3)
                merged.append(item)
                continue
            # Heavy overlap: glue into one continuous highlight.
            gs = min(m.get("g_start", m["start"]), item.get("g_start", item["start"]))
            ge = max(m.get("g_end", m["end"]), item.get("g_end", item["end"]))
            m["g_start"], m["g_end"] = gs, ge
            m["start"], m["end"] = round(gs - off, 3), round(ge - off, 3)
            m["event_times"] = sorted(set(m["event_times"] + item["event_times"]))
            m["visual_count"] += item["visual_count"]
            m["audio_count"] += item["audio_count"]
            if item.get("confirmed_detail"):
                seen = {k["t"] for k in m.get("confirmed_detail", [])}
                m["confirmed_detail"] = m.get("confirmed_detail", []) + [
                    k for k in item["confirmed_detail"] if k["t"] not in seen]
                m["confirmed_detail"].sort(key=lambda k: k["t"])
            if item.get("ocr_kills"):
                seen = {(k["t"], k["killer"], k["victim"]) for k in m.get("ocr_kills", [])}
                m["ocr_kills"] = m.get("ocr_kills", []) + [
                    k for k in item["ocr_kills"]
                    if (k["t"], k["killer"], k["victim"]) not in seen]
                m["ocr_kills"].sort(key=lambda k: k["t"])
            if item.get("card_kills"):
                seen = {k["t"] for k in m.get("card_kills", [])}
                m["card_kills"] = m.get("card_kills", []) + [
                    k for k in item["card_kills"] if k["t"] not in seen]
                m["card_kills"].sort(key=lambda k: k["t"])
            m["peak_strength"] = max(m["peak_strength"], item["peak_strength"])
            m["score"] = round(max(m["score"], item["score"]), 4)
            m["merged_from"] = m.get("merged_from", 1) + 1
            if m["end"] - m["start"] > max_len:
                m["start"] = round(m["end"] - max_len, 3)
                # 同素材内 g 与局部时间差恒为 off：按局部时长同步全局窗口
                m["g_start"] = round(m.get("g_end", m["end"]) - max_len, 3)
            m["duration"] = round(m["end"] - m["start"], 3)
            # 截断窗口后重算：事件/确认击杀只统计最终窗口内的，否则被截掉
            # 前段的卡牌击杀会虚增 kill_count 下限（实测合并段多算 1 杀）
            m["event_times"] = [t for t in m["event_times"]
                                if m["start"] <= t <= m["end"]]
            m["kill_times"] = [t for t in m.get("kill_times", [])
                               if m["start"] <= t <= m["end"]]
            detail = [k for k in m.get("confirmed_detail", [])
                      if m["start"] - 1.0 <= k["t"] <= m["end"]]
            m["confirmed_detail"] = detail
            m["confirmed_kills"] = sum(k["kills"] for k in detail)
            m["card_kills"] = [k for k in (m.get("card_kills") or [])
                               if m["start"] - 1.0 <= k["t"] <= m["end"]]
        else:
            merged.append(item)
    chosen = merged
    total = sum(c["duration"] for c in chosen)

    for rank, c in enumerate(chosen, 1):
        c["rank"] = rank
        if c.get("merged_from", 1) > 1:
            c["reason"] += f"；已合并 {c['merged_from']} 段相邻片段"
        c["caption"] = f"HIGHLIGHT {rank:02d}  {c['reason']}"

    multi = len(sources) > 1
    return {
        "source_video": [s["video"] for s in sources] if multi else str(video),
        "sources": [{"index": i, "video": s["video"],
                     "offset": round(s["offset"], 3),
                     "duration": round(s["duration"], 3)}
                    for i, s in enumerate(sources)],
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "duration": round(duration, 3),
        "event_count": {"visual": len(visual_events), "audio": len(audio_events)},
        "cluster_count": len(candidates),
        "selected_count": len(chosen),
        "total_selected_sec": round(total, 3),
        "config": {"roi": cfg["roi"], "sample": cfg["sample"], "detect": cfg["detect"],
                   "recognizer": cfg.get("recognizer", {"backend": "none"}),
                   "ocr": {k: v for k, v in cfg.get("ocr", {}).items() if k != "_comment"},
                   "clip": cfg["clip"], "select": cfg["select"], "render": cfg["render"]},
        "candidates": candidates,
        "selected": chosen,
    }
