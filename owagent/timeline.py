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


def build_timeline(video: str | Path, visual_events: list[dict], audio_events: list[dict],
                   duration: float, cfg: dict, ocr: dict | None = None) -> dict:
    clip = cfg["clip"]
    sel = cfg["select"]
    pre, post = float(clip["pre_roll"]), float(clip["post_roll"])
    gap, min_len, max_len = float(clip["merge_gap"]), float(clip["min_len"]), float(clip["max_len"])

    all_events = sorted(visual_events + audio_events, key=lambda e: e["t"])
    clusters: list[dict] = []
    for e in all_events:
        if clusters and e["t"] - clusters[-1]["last"] <= gap:
            c = clusters[-1]
            c["events"].append(e)
            c["last"] = e["t"]
        else:
            clusters.append({"events": [e], "first": e["t"], "last": e["t"]})

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
        # 骷髅头/击杀卡牌确认 = 录像玩家的击杀（这些 HUD 只对击杀者本人显示），
        # 是比音频佐证更强的多通道证据；killcard 事件带 kills 增量（一跳多杀）
        confirmed = [e for e in events if e.get("confirmed")]
        confirmed_kills = sum(e.get("kills", 1) for e in confirmed)
        base = max((e["strength"] for e in events), default=0.0)
        combo = min(len(vis) / 4.0, 1.0) if vis else min(len(aud) / 3.0, 1.0) * 0.5
        cross = 1.0 if (vis and aud) else 0.0
        margin = 1.0 if (start >= 5.0 and end <= duration - 5.0) else 0.6
        # score = 0.40*peak + 0.30*combo + 0.20*cross-modal + 0.10*context
        #         (+0.10 玩家击杀确认，每多一杀 +0.05，封顶 0.20)
        kill_bonus = min(0.10 + 0.05 * max(confirmed_kills - 1, 0), 0.20) if confirmed_kills else 0.0
        score = (0.40 * base + 0.30 * combo + 0.20 * cross + 0.10 * margin
                 + kill_bonus)

        # OCR 读到的具名击杀（含连杀数），供 VLM 复核提示与 L3 融合使用
        ocr_kills = [{"t": e["t"], "killer": e.get("killer", ""),
                      "victim": e.get("victim", ""), "streak": e.get("streak", 1)}
                     for e in events if e.get("killer")]
        max_streak = max((k["streak"] for k in ocr_kills), default=0)
        # 击杀卡牌确认的玩家击杀（增量与本回合累计数），供 VLM 复核使用
        card_kills = [{"t": e["t"], "kills": e.get("kills", 1),
                       "streak": e.get("streak", 1)}
                      for e in events if "killcard" in (e.get("channels") or [])]

        bits = []
        card_events = [e for e in events if "killcard" in (e.get("channels") or [])]
        if card_events:
            top = max(e.get("streak", 1) for e in card_events)
            bits.append(f"击杀卡牌确认玩家击杀（本回合累计 {top} 杀）")
        elif confirmed:
            killer = confirmed[0].get("killer") or "玩家"
            bits.append(f"骷髅头确认玩家击杀×{len(confirmed)}（{killer}）")
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
        candidates.append({
            "id": idx,
            "start": round(start, 3), "end": round(end, 3),
            "duration": round(end - start, 3),
            "event_times": [round(e["t"], 3) for e in events],
            "kill_times": [round(e["t"], 3) for e in vis],
            "visual_count": len(vis), "audio_count": len(aud),
            "ocr_kills": ocr_kills,
            "card_kills": card_kills,
            "peak_strength": round(base, 4),
            "score": round(score, 4),
            "reason": reason,
            "caption": f"HIGHLIGHT {idx:02d}  " + (" / ".join(bits[:2]) or "候选高光"),
            "selected": False,
        })

    for wv in waves:
        first = min(e["t"] for e in wv)
        last = max(e["t"] for e in wv)
        start = max(0.0, first - pre)
        end = min(duration, last + post)
        if end - start < min_len:
            end = min(duration, start + min_len)
        end = min(end, duration)
        if end - start < 0.5:
            continue

        if end - start <= max_len:
            _emit(wv, start, end)
            continue

        # 超长波：按视觉事件密度产出多个 max_len 窗口（消耗式切分），
        # 每个窗口独立打分为候选，避免一波长战斗只保留 16 秒。
        # 上限 5：一波 76s 的战斗可能有 3~4 个密集击杀区（前中后段），
        # 太少会让中后段的三杀窗口永远轮不上。
        vis_ts = sorted(e["t"] for e in wv if e["source"] in VIS_SOURCES)
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
            _emit([e for e in wv if ws <= e["t"] <= we], ws, we)

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
    chosen.sort(key=lambda c: c["start"])  # final reel stays in chronological order
    chosen_ids = {c["id"] for c in chosen}
    for c in candidates:
        c["selected"] = c["id"] in chosen_ids

    # Overlapping clips would repeat the same frames in the final reel.
    merged: list[dict] = []
    for c in chosen:
        item = dict(c)
        if merged and item["start"] < merged[-1]["end"]:
            m = merged[-1]
            overlap = m["end"] - item["start"]
            if overlap <= 2.0 and (item["end"] - m["end"]) >= min_len * 0.5:
                # Small overlap: trim the later clip, keep it as a separate shot.
                item["start"] = round(m["end"], 3)
                item["duration"] = round(item["end"] - item["start"], 3)
                merged.append(item)
                continue
            # Heavy overlap: glue into one continuous highlight.
            m["start"] = min(m["start"], item["start"])
            m["end"] = max(m["end"], item["end"])
            m["event_times"] = sorted(set(m["event_times"] + item["event_times"]))
            m["visual_count"] += item["visual_count"]
            m["audio_count"] += item["audio_count"]
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
                m["start"] = m["end"] - max_len
            m["duration"] = round(m["end"] - m["start"], 3)
        else:
            merged.append(item)
    chosen = merged
    total = sum(c["duration"] for c in chosen)

    for rank, c in enumerate(chosen, 1):
        c["rank"] = rank
        if c.get("merged_from", 1) > 1:
            c["reason"] += f"；已合并 {c['merged_from']} 段相邻片段"
        c["caption"] = f"HIGHLIGHT {rank:02d}  {c['reason']}"

    return {
        "source_video": str(video),
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
