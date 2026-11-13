"""L2/L3: merge recalled events into candidate clips, score, select Top-K."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def build_timeline(video: str | Path, visual_events: list[dict], audio_events: list[dict],
                   duration: float, cfg: dict) -> dict:
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

    candidates = []
    for i, c in enumerate(clusters):
        start = max(0.0, c["first"] - pre)
        end = min(duration, c["last"] + post)
        if end - start < min_len:
            end = min(duration, start + min_len)
        if end - start > max_len:
            start = end - max_len  # keep the climax, trim the build-up
        end = min(end, duration)
        if end - start < 0.5:
            continue

        vis = [e for e in c["events"] if e["source"] in ("visual", "model")]
        aud = [e for e in c["events"] if e["source"] == "audio"]
        mdl = [e for e in c["events"] if e["source"] == "model"]
        base = max((e["strength"] for e in c["events"]), default=0.0)
        combo = min(len(vis) / 4.0, 1.0) if vis else min(len(aud) / 3.0, 1.0) * 0.5
        cross = 1.0 if (vis and aud) else 0.0
        margin = 1.0 if (start >= 5.0 and end <= duration - 5.0) else 0.6
        # score = 0.40*peak + 0.30*combo + 0.20*cross-modal + 0.10*context
        score = 0.40 * base + 0.30 * combo + 0.20 * cross + 0.10 * margin

        bits = []
        if mdl:
            bits.append(f"{len(mdl)} 次模型判定击杀"
                        + (f"（最高 p={max(e['strength'] for e in mdl):.2f}）" if mdl else ""))
        elif vis:
            bits.append(f"{len(vis)} 次播报")
        if aud:
            bits.append(f"{len(aud)} 次音频峰值")
        if len(vis) >= 3:
            bits.append("疑似连杀")
        if cross:
            bits.append("视听双通道确认")
        reason = "、".join(bits) + f"，峰值强度 {base:.2f}"

        candidates.append({
            "id": i + 1,
            "start": round(start, 3), "end": round(end, 3),
            "duration": round(end - start, 3),
            "event_times": [round(e["t"], 3) for e in c["events"]],
            "visual_count": len(vis), "audio_count": len(aud),
            "peak_strength": round(base, 4),
            "score": round(score, 4),
            "reason": reason,
            "caption": f"HIGHLIGHT {i + 1:02d}  " + (" / ".join(bits[:2]) or "候选高光"),
            "selected": False,
        })

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
                   "clip": cfg["clip"], "select": cfg["select"], "render": cfg["render"]},
        "candidates": candidates,
        "selected": chosen,
    }
