"""L2 semantic review: a vision-language model inspects keyframes of each
candidate highlight and answers "what happened & how exciting was it?"

Design:
  * Only the ALREADY-SELECTED segments (typically <= 8) are reviewed, a few
    keyframes each - a 30-min match costs a handful of VLM calls, not 43k.
  * Backend is any OpenAI-compatible chat API with image input
    (default: DeepSeek deepseek-flash, input_modalities=[text, image]).
  * The model returns strict JSON per segment: event type, kill count,
    excitement 0-10, one-line commentary, confidence.
  * Results blend into the final score (L3) and non-highlights can be
    dropped - but ANY failure (no key, network, timeout, bad JSON) simply
    leaves the pipeline untouched. Review never blocks the reel.

API key resolution order:
  1. env var  review.api_key_env  (default OWAGENT_REVIEW_KEY)
  2. local file review.api_key_file (default config.local.json, gitignored)
     with field "review_api_key"
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .util import ToolError, log, run_ffmpeg

ROOT = Path(__file__).resolve().parent.parent

EVENT_WEIGHTS = {
    "team_wipe": 1.00,   # 团灭
    "multi_kill": 0.90,  # 连杀（>=2）
    "ultimate": 0.80,    # 大招时刻
    "rescue": 0.70,      # 关键救人/守住
    "objective": 0.60,   # 推进/占点关键节点
    "single_kill": 0.50,
    "other": 0.40,
    "not_game": 0.00,    # 根本不是对局画面（误检）
}
EVENT_NAMES = {
    "team_wipe": "团灭", "multi_kill": "连杀", "ultimate": "大招时刻",
    "rescue": "关键救援", "objective": "目标点攻防", "single_kill": "单杀",
    "other": "其他事件", "not_game": "非对局画面",
}

SYSTEM_PROMPT = """你是《守望先锋2》(Overwatch 2) 集锦审片员，精通其 HUD：
- 击杀播报(kill feed)位于画面**右上角**：击杀者名 + 武器图标 + 被击杀者名，短时间多行即连杀；
- 画面**中央偏下**的大字播报为团队事件：TEAM KILL(团灭)、英雄名+大招连招提示等；
- 屏幕下方中央是血条/技能冷却；左下角是队友状态；顶部中央是比分/时间；
- 死亡回放(death replay)与观战视角没有第一人称 HUD，注意区分。
你将看到同一段录像按时间顺序排列的若干关键帧。请判断这段画面里发生了什么，输出**严格 JSON**（不要多余文字）：
{"is_highlight": bool, "event_type": "team_wipe|multi_kill|ultimate|rescue|objective|single_kill|other|not_game",
 "kill_count": int, "excitement": 0到10的整数, "confidence": 0到1的小数, "commentary": "不超过20字的中文解说"}
评分口径：团灭/三杀以上 8-10；双杀或关键大招 6-8；普通单杀 3-5；平淡画面或非对局 0-2。
kill_count 不确定就填 0。confidence 表示你对判断的把握。"""


class ReviewNotReady(ToolError):
    """The review layer cannot run (no key / bad config)."""


# ---------------------------------------------------------------- keyframes
def _pick_times(seg: dict, n: int) -> list[float]:
    """Pick up to n representative timestamps inside a segment."""
    start, end = float(seg["start"]), float(seg["end"])
    events = sorted(set(float(t) for t in seg.get("event_times", [])))
    picks = {min(end - 0.2, start + 0.5), max(start + 0.2, end - 0.5)}
    if events:
        picks.add(min(max(events[len(events) // 2], start), end))
        picks.add(min(max(events[0], start), end))
        picks.add(min(max(events[-1], start), end))
    picks = {min(max(t, start + 0.05), end - 0.05) for t in picks}
    return sorted(picks)[:n]


def _extract_frame(video: Path, t: float, width: int) -> bytes:
    """Decode one frame at time t as JPEG bytes via ffmpeg pipe."""
    data = run_ffmpeg(
        ["-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-2:flags=bilinear", "-q:v", "5",
         "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
        capture=True)
    if not data:
        raise ToolError(f"抽帧失败: t={t:.2f}")
    return data


# ---------------------------------------------------------------- API
def _resolve_key(cfg: dict) -> str:
    rv = cfg.get("review", {})
    key = os.environ.get(rv.get("api_key_env", "OWAGENT_REVIEW_KEY"), "")
    if not key:
        f = Path(rv.get("api_key_file") or (ROOT / "config.local.json"))
        if not f.is_absolute():
            f = ROOT / f
        if f.is_file():
            try:
                key = (json.loads(f.read_text(encoding="utf-8"))
                       .get("review_api_key", "")).strip()
            except (ValueError, OSError):
                key = ""
    return key


def ping_api(cfg: dict, key: str | None = None) -> tuple[bool, str]:
    """Cheap pre-flight check: connectivity + key validity + model availability.

    Calls GET {base_url}/models (no tokens consumed). Returns (ok, message).
    """
    key = (key or _resolve_key(cfg) or "").strip()
    if not key:
        return False, "未配置 API Key"
    rv = cfg.get("review", {})
    url = rv.get("base_url", "https://api.deepseek.com").rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", "?") for m in data.get("data", [])]
        model = rv.get("model", "deepseek-flash")
        if models and model not in models:
            return False, f"Key 有效，但配置的模型 {model} 不在可用列表（{', '.join(models[:5])}…）"
        return True, f"连接正常（{len(models)} 个模型可用，含 {model}）"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "API Key 无效或已过期（401）"
        return False, f"服务端返回 HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - network layer, any error is a check failure
        return False, f"网络错误: {str(e)[:120]}"


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply (may have fences/prose)."""
    if not text:
        return {}
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _norm_review(obj: dict) -> dict:
    """Validate & clamp the model's JSON into a predictable shape."""
    et = str(obj.get("event_type", "other")).strip().lower()
    if et not in EVENT_WEIGHTS:
        et = "other"
    try:
        kills = max(0, min(99, int(obj.get("kill_count") or 0)))
    except (TypeError, ValueError):
        kills = 0
    try:
        exc = max(0, min(10, int(round(float(obj.get("excitement") or 0)))))
    except (TypeError, ValueError):
        exc = 0
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence") or 0.5)))
    except (TypeError, ValueError):
        conf = 0.5
    is_hl = obj.get("is_highlight")
    if not isinstance(is_hl, bool):
        is_hl = bool(exc >= 4 and et not in ("not_game",))
    if kills >= 3 and et in ("single_kill", "other"):
        et, is_hl = "multi_kill", True
    elif kills == 2 and et in ("single_kill", "other"):
        et = "multi_kill"
    if et == "team_wipe":
        is_hl = True
    comment = str(obj.get("commentary") or "").strip().replace("\n", " ")[:40]
    return {"is_highlight": is_hl, "event_type": et, "kill_count": kills,
            "excitement": exc, "confidence": round(conf, 2),
            "event_label": EVENT_NAMES[et], "commentary": comment}


def _call_vlm(video: Path, seg: dict, cfg: dict, key: str) -> dict:
    """One API call: N keyframes of one segment -> normalized review dict."""
    rv = cfg["review"]
    frames = []
    for t in _pick_times(seg, int(rv.get("frames_per_segment", 4))):
        frames.append((t, _extract_frame(video, t, int(rv.get("frame_width", 768)))))
    if not frames:
        raise ToolError("没有可用的关键帧")

    content = [{
        "type": "text",
        "text": (f"片段 #{seg.get('rank', seg.get('id', '?'))}，"
                 f"源片 {seg['start']:.1f}s - {seg['end']:.1f}s。"
                 f"下面 {len(frames)} 张是按时间顺序的关键帧，判断后输出 JSON。"),
    }]
    for _, jpg in frames:
        b64 = base64.b64encode(jpg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": rv.get("model", "deepseek-flash"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content}],
        "max_tokens": int(rv.get("max_tokens", 3000)),
        "temperature": 0.2,
    }
    effort = rv.get("effort")
    if effort:
        payload["reasoning_effort"] = effort
    url = rv.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=float(rv.get("timeout_sec", 180))) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    usage = data.get("usage", {})
    text = data["choices"][0]["message"].get("content") or ""
    out = _norm_review(_extract_json(text))
    out["tokens"] = usage.get("total_tokens", 0)
    out["frames"] = len(frames)
    return out


# ---------------------------------------------------------------- pipeline
def _blended_score(seg: dict, review: dict, rv: dict) -> float:
    """Fuse VLM semantics with the signal score:  w_s*signal + w_v*(exc/10) + w_e*event."""
    ws = float(rv.get("w_signal", 0.35))
    wv = float(rv.get("w_vlm", 0.45))
    we = float(rv.get("w_event", 0.20))
    ev = EVENT_WEIGHTS.get(review["event_type"], 0.4)
    return round(ws * float(seg["score"]) + wv * (review["excitement"] / 10.0)
                 + we * ev, 4)


def review_timeline(video: str | Path, tl: dict, cfg: dict,
                     work_dir: str | Path | None = None) -> dict:
    """Review the selected segments of a timeline in place. Returns a summary."""
    video = Path(video)
    rv = cfg.get("review") or {}
    if not rv.get("enabled"):
        return {"skipped": "disabled"}
    key = _resolve_key(cfg)
    if not key:
        raise ReviewNotReady(
            "未找到 VLM API Key：请设置环境变量 "
            f"{rv.get('api_key_env', 'OWAGENT_REVIEW_KEY')}，或在 "
            f"{rv.get('api_key_file', 'config.local.json')} 里写 {{\"review_api_key\": \"sk-...\"}}")

    selected = [s for s in tl.get("selected", [])]
    if not selected:
        log("没有入选片段，跳过语义复核", "review")
        return {"skipped": "no segments"}
    max_seg = int(rv.get("max_segments", 8))
    todo = selected[:max_seg]
    log(f"语义复核：{len(todo)}/{len(selected)} 段，"
        f"每段最多 {rv.get('frames_per_segment', 4)} 关键帧，"
        f"模型 {rv.get('model', 'deepseek-flash')}（{rv.get('provider', 'deepseek')} 云端）", "review")

    def work(seg):
        return seg, _call_vlm(video, seg, cfg, key)

    results: dict[int, dict] = {}
    errors: list[str] = []
    workers = max(1, min(4, int(rv.get("concurrency", 2))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, s) for s in todo]
        for fut in as_completed(futs):
            try:
                seg, review = fut.result()
                results[seg["id"]] = review
                log(f"  片段 #{seg.get('rank', seg['id'])}: {review['event_label']}"
                    f" exc={review['excitement']}/10 kills={review['kill_count']}"
                    f" conf={review['confidence']:.2f} 「{review['commentary']}」", "review")
            except Exception as exc:  # noqa: BLE001 - per-segment isolation
                errors.append(str(exc))
                log(f"  某段复核失败（不影响其他段）: {str(exc)[:120]}", "warn")

    dropped: list[dict] = []
    total_tokens = 0
    for seg in selected:
        review = results.get(seg["id"])
        if review is None:
            seg["review"] = {"error": "review_failed"}
            continue
        total_tokens += review.get("tokens", 0)
        seg["review"] = review
        seg["score_signal"] = seg["score"]
        seg["score"] = _blended_score(seg, review, rv)
        seg["reason"] = (f"{review['event_label']}"
                         + (f"×{review['kill_count']}" if review["kill_count"] else "")
                         + f"、VLM精彩 {review['excitement']}/10"
                         + f"（{seg['reason']}）")
        if review.get("commentary"):
            seg["caption"] = f"HIGHLIGHT {seg.get('rank', 1):02d}  {review['commentary']}"
            seg["llm_caption"] = review["commentary"]

    if rv.get("drop_non_highlight", True):
        keep, drop = [], []
        for seg in selected:
            review = seg.get("review") or {}
            # 只有当模型明确否认且置信度足够时才丢弃，低置信度保留
            if (review.get("is_highlight") is False
                    and float(review.get("confidence", 0)) >= 0.7):
                drop.append(seg)
            else:
                keep.append(seg)
        if drop and keep:  # never drop everything
            cand_by_id = {c["id"]: c for c in tl.get("candidates", [])}
            for seg in drop:
                seg["selected"] = False
                seg["drop_reason"] = f"VLM 判定非高光（{seg['review'].get('event_label', '')}）"
                log(f"  丢弃片段 #{seg.get('rank')}: {seg['drop_reason']}", "review")
                # selected 持有的是 candidates 的副本，需同步回原对象
                orig = cand_by_id.get(seg["id"])
                if orig is not None:
                    orig["selected"] = False
                    orig["drop_reason"] = seg["drop_reason"]
                    orig["review"] = seg.get("review")
            dropped = drop
            selected = keep

    # re-rank surviving segments & refresh counters
    selected.sort(key=lambda s: s["start"])
    for rank, seg in enumerate(selected, 1):
        seg["rank"] = rank
        if seg.get("review", {}).get("commentary"):
            seg["caption"] = f"HIGHLIGHT {rank:02d}  {seg['review']['commentary']}"
    tl["selected"] = selected
    tl["selected_count"] = len(selected)
    tl["total_selected_sec"] = round(sum(s["duration"] for s in selected), 3)
    tl["review_summary"] = {
        "provider": rv.get("provider", "deepseek"),
        "model": rv.get("model", "deepseek-flash"),
        "reviewed": len(results), "failed": len(errors),
        "dropped": len(dropped), "total_tokens": total_tokens,
    }
    log(f"复核完成：成功 {len(results)} / 失败 {len(errors)} / 丢弃 {len(dropped)} 段，"
        f"消耗约 {total_tokens} tokens", "review")
    return tl["review_summary"]
