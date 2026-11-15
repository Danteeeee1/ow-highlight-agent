"""Agent orchestration layer: decisions, optional LLM captions, human-readable report.

The agent only consumes structured events / the timeline JSON - never raw video frames.
This keeps token cost flat regardless of recording length, and degrades gracefully
when no API key or network is available.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .util import fmt_time, log


def _llm_captions(selected: list[dict], cfg: dict) -> list[str] | None:
    a = cfg["agent"]
    key = os.environ.get(a["api_key_env"], "")
    if not key:
        log(f"未设置环境变量 {a['api_key_env']}，跳过 LLM 解说，使用本地文案", "agent")
        return None
    game_names = {"overwatch": "守望先锋", "csgo2": "CS2", "valorant": "瓦罗兰特"}
    game = game_names.get(cfg.get("recognizer", {}).get("game", "overwatch"), "FPS")
    compact = [{"no": s.get("rank"), "start": s["start"], "duration": s["duration"],
                "score": s["score"], "reason": s["reason"]} for s in selected]
    payload = {
        "model": a["model"],
        "messages": [
            {"role": "system", "content": f"你是《{game}》集锦剪辑助理。根据结构化片段信息，为每段写一句不超过 25 字的中文解说词。只输出纯文本，每行一段，行数与片段数一致，不要编号、不要多余说明。"},
            {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
        ],
        "temperature": 0.7,
    }
    url = a["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=float(a["timeout_sec"])) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        lines = [ln.strip(" -0123456789.\t") for ln in text.splitlines() if ln.strip()]
        if lines:
            log(f"LLM 生成 {len(lines)} 条解说词", "agent")
            return lines[:len(selected)]
    except (urllib.error.URLError, KeyError, ValueError, OSError) as exc:
        log(f"LLM 调用失败，自动降级到本地文案: {exc}", "warn")
    return None


def _advice(tl: dict) -> list[str]:
    out: list[str] = []
    cands = tl["candidates"]
    if not cands:
        out.append("一个候选片段都没有：先跑 `roi-preview` 确认 ROI 框住击杀播报，再把 `--threshold` 降到 0.30 左右。")
    elif tl["selected_count"] == 0:
        out.append("有候选但没入选：检查 `select.min_score`，或降低 `--threshold`。")
    avg = sum(c["score"] for c in cands) / len(cands) if cands else 0
    if avg < 0.35:
        out.append(f"平均得分仅 {avg:.2f}：多半是 ROI 没对准或画面与默认配置差异大，建议重新标定 ROI。")
    if len(cands) > 40:
        out.append(f"候选多达 {len(cands)} 段，误检可能偏多：提高 `--threshold`，或增大 `--gap` 让相邻事件合并。")
    vis = tl["event_count"]["visual"]
    aud = tl["event_count"]["audio"]
    if aud == 0:
        out.append("没有召回音频峰值：录像可能无音轨，或 `audio.peak_ratio` 过高；音轨是低成本的交叉验证信号。")
    if vis and aud and vis / max(aud, 1) > 6:
        out.append("视觉事件远多于音频事件：可略降 `audio.peak_ratio` 增强双通道确认。")
    if tl.get("config", {}).get("recognizer", {}).get("backend") == "crispy":
        out.append("Crispy 模型召回已启用：无检出时把 `--confidence` 降到 0.4~0.5；"
                   "误检多则升到 0.7~0.8；并确认录像为 16:9 全屏（模型按 1920x1080 训练）。")
    else:
        out.append("P0 只做视觉/音频信号召回，不认识英雄与击杀语义；"
                    "要提质请按 README「模型识别」一节启用 Crispy 神经网络（--recognizer crispy）。")
    if tl.get("review_summary"):
        out.append("语义复核已启用：误杀过多把 `review.w_vlm` 降到 0.35 或 `drop_non_highlight` 关掉；"
                   "嫌慢把 `review.concurrency` 提到 3-4 或 `frames_per_segment` 降到 3；"
                   "解说不满意可重跑（模型输出有随机性，temperature 已设 0.2）。")
    return out


def orchestrate(tl: dict, cfg: dict, work_dir: str | Path, render_info: dict | None = None) -> dict:
    work_dir = Path(work_dir)
    selected = tl["selected"]
    review_sum = tl.get("review_summary") or {}
    # VLM 复核已产出解说词时不再调纯文本 LLM 重复生成
    has_review_caps = any(s.get("llm_caption") for s in selected)
    captions = None
    if cfg["agent"]["llm_enabled"] and not has_review_caps:
        captions = _llm_captions(selected, cfg)
    llm_used = bool(captions)
    for i, seg in enumerate(selected):
        if captions and i < len(captions):
            seg["llm_caption"] = captions[i]

    lines: list[str] = []
    rec = cfg.get("recognizer", {})
    backend = rec.get("backend", "none")
    lines.append("# 高光剪辑决策报告\n")
    lines.append(f"- 源视频：`{tl['source_video']}`")
    lines.append(f"- 时长：{fmt_time(tl['duration'])}")
    if backend == "crispy":
        from .reviewer import GAME_NAMES
        game = rec.get("game", "overwatch")
        lines.append(f"- 识别后端：Crispy 神经网络（{GAME_NAMES.get(game, game)}，"
                     f"confidence={rec.get('confidence')}，{tl['event_count']['visual']} 次击杀判定）")
        lines.append(f"- 召回事件：模型击杀 {tl['event_count']['visual']} 个 / 音频 {tl['event_count']['audio']} 个")
    else:
        lines.append(f"- 识别后端：启发式（像素信号）")
        lines.append(f"- 召回事件：视觉 {tl['event_count']['visual']} 个 / 音频 {tl['event_count']['audio']} 个")
    lines.append(f"- 候选片段：{tl['cluster_count']} 段，入选 **{tl['selected_count']}** 段，合计 {tl['total_selected_sec']:.1f}s")
    if review_sum:
        lines.append(f"- 语义复核：{review_sum.get('model')}（{review_sum.get('provider')} 云端），"
                     f"成功 {review_sum.get('reviewed')} 段 / 丢弃 {review_sum.get('dropped')} 段 / "
                     f"失败 {review_sum.get('failed')} 段，约 {review_sum.get('total_tokens')} tokens")
    if render_info:
        lines.append(f"- 成片：`{render_info['output']}`（{render_info['resolution']}，字幕烧录 {'是' if render_info['subtitles_burned'] else '否'}）")
    else:
        lines.append("- 成片：未渲染（dry-run 模式）")
    if review_sum:
        lines.append(f"- 解说词来源：VLM 复核（{review_sum.get('model')}）\n")
    else:
        lines.append(f"- 解说词来源：{'LLM（' + cfg['agent']['model'] + '）' if llm_used else '本地规则文案'}\n")

    lines.append("## 入选片段（按成片时序）\n")
    if selected:
        if review_sum:
            lines.append("| # | 源片时间 | 时长 | 综合分 | 事件 | VLM精彩 | 解说词 |")
            lines.append("|---|---|---|---|---|---|---|")
            for seg in selected:
                rv = seg.get("review") or {}
                cap = rv.get("commentary") or seg.get("llm_caption") or "—"
                lines.append(f"| {seg.get('rank')} | {fmt_time(seg['start'])} → {fmt_time(seg['end'])} | "
                             f"{seg['duration']:.1f}s | {seg['score']:.3f} | "
                             f"{rv.get('event_label', '—')}"
                             + (f"×{rv['kill_count']}" if rv.get("kill_count") else "")
                             + f" | {rv.get('excitement', '—')}/10 | {cap} |")
        else:
            lines.append("| # | 源片时间 | 时长 | 得分 | 判定理由 | 解说词 |")
            lines.append("|---|---|---|---|---|---|")
            for seg in selected:
                cap = seg.get("llm_caption") or "—"
                lines.append(f"| {seg.get('rank')} | {fmt_time(seg['start'])} → {fmt_time(seg['end'])} | "
                             f"{seg['duration']:.1f}s | {seg['score']:.3f} | {seg['reason']} | {cap} |")
    else:
        lines.append("_无入选片段_")

    rejected = sorted((c for c in tl["candidates"] if not c["selected"]),
                      key=lambda c: c["score"], reverse=True)[:5]
    lines.append("\n## 落选候选（得分最高的 5 段）\n")
    if rejected:
        lines.append("| 源片时间 | 得分 | 理由 |")
        lines.append("|---|---|---|")
        for c in rejected:
            extra = f"（{c['drop_reason']}）" if c.get("drop_reason") else ""
            lines.append(f"| {fmt_time(c['start'])} | {c['score']:.3f} | {c['reason']}{extra} |")
    else:
        lines.append("_无_")

    lines.append("\n## 打分公式\n")
    if review_sum:
        from .reviewer import event_names, event_weights
        rgame = review_sum.get("game") or cfg.get("recognizer", {}).get("game", "overwatch")
        lines.append("`综合分 = w_signal × 信号分 + w_vlm × (VLM精彩/10) + w_event × 事件权重`\n")
        lines.append(f"- 权重：w_signal={cfg.get('review', {}).get('w_signal')} / "
                     f"w_vlm={cfg.get('review', {}).get('w_vlm')} / "
                     f"w_event={cfg.get('review', {}).get('w_event')}")
        ew = event_weights(rgame)
        en = event_names(rgame)
        chain = " > ".join(f"{en[k]} {ew[k]:.2f}" for k in
                           sorted(ew, key=ew.get, reverse=True) if k != "not_game")
        lines.append(f"- 事件权重（{rgame}）：{chain}")
        lines.append("- VLM 明确判定非高光且置信度 ≥ 0.7 的段会被丢弃（可在 review.drop_non_highlight 关闭）")
    else:
        lines.append("`score = 0.40 × 峰值强度 + 0.30 × 连击度 + 0.20 × 视听交叉确认 + 0.10 × 上下文`\n")
    lines.append("- 连击度 = min(同一窗口内视觉事件数 / 4, 1)")
    lines.append("- 视听交叉确认 = 同窗口内同时有视觉与音频事件时为 1")
    lines.append("- 上下文 = 避开片头/片尾 5s 时为 1，否则 0.6")
    lines.append("- 选段后按**源片时间顺序**重排，集锦叙事才连贯\n")

    lines.append("## 当前参数快照\n")
    lines.append("```json")
    lines.append(json.dumps({"roi": cfg["roi"], "sample": cfg["sample"],
                             "recognizer": cfg.get("recognizer", {"backend": "none"}),
                             "detect": cfg["detect"], "clip": cfg["clip"],
                             "select": cfg["select"]}, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## 调参建议\n")
    for item in _advice(tl):
        lines.append(f"- {item}")

    lines.append("\n## 产物清单\n")
    lines.append("| 文件 | 用途 |")
    lines.append("|---|---|")
    lines.append("| `timeline.json` | 全部候选 + 入选标记，Agent 的状态快照 |")
    lines.append("| `signals.csv` | 每一采样帧的三路信号与得分，调阈值时看它 |")
    lines.append("| `audio_rms.csv` | 音频能量包络 |")
    lines.append("| `captions.srt` | 按成片时序的字幕，可直接拖进播放器或剪辑软件 |")
    lines.append("| `highlights.mp4` | 成片 |")
    lines.append("| `parts/` | 各切片中间文件，便于排查 |")

    report = work_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"决策报告已生成: {report}", "agent")
    return {"report": str(report), "llm_used": llm_used}
