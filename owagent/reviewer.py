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

# ---------------------------------------------------------------- per-game
# 每游戏一套完整事件词表：{event_type: (中文名, 权重)}。
# 词表同时决定 VLM 提示词中的事件枚举、结果校验与 L3 融合打分。
GAME_EVENTS = {
    "overwatch": {
        "team_wipe": ("团灭", 1.00),
        "multi_kill": ("连杀", 0.90),
        "ultimate": ("大招时刻", 0.80),
        "rescue": ("关键救援", 0.70),
        "objective": ("目标点攻防", 0.60),
        "single_kill": ("单杀", 0.50),
        "other": ("其他事件", 0.40),
        "not_game": ("非对局画面", 0.00),
    },
    "csgo2": {
        "ace": ("五杀ACE", 1.00),
        "clutch": ("残局", 0.95),
        "multi_kill": ("连杀", 0.90),
        "defuse": ("拆弹成功", 0.75),
        "objective": ("回合攻防", 0.60),
        "single_kill": ("单杀", 0.50),
        "other": ("其他事件", 0.40),
        "not_game": ("非对局画面", 0.00),
    },
    "valorant": {
        "ace": ("五杀ACE", 1.00),
        "clutch": ("残局", 0.95),
        "multi_kill": ("连杀", 0.90),
        "ultimate": ("大招时刻", 0.80),
        "defuse": ("拆弹成功", 0.75),
        "spike": ("爆能器攻防", 0.60),
        "single_kill": ("单杀", 0.50),
        "other": ("其他事件", 0.40),
        "not_game": ("非对局画面", 0.00),
    },
}
GAME_NAMES = {"overwatch": "守望先锋2", "csgo2": "CS2", "valorant": "瓦罗兰特"}

# 模型输出跨游戏词时的兜底映射（该词不在当前游戏词表时才启用）
_CROSS_GAME_ALIAS = {
    "team_wipe": "ace", "ace": "team_wipe",
    "clutch": "multi_kill", "defuse": "objective", "spike": "objective",
    "ultimate": "other", "rescue": "objective",
}


def _game_table(game: str) -> dict:
    return GAME_EVENTS.get(game, GAME_EVENTS["overwatch"])


def event_weights(game: str) -> dict:
    return {k: v[1] for k, v in _game_table(game).items()}


def event_names(game: str) -> dict:
    return {k: v[0] for k, v in _game_table(game).items()}


# 各游戏 HUD 知识 + 评分口径，喂给 VLM 的 system prompt 由它拼装。
GAME_KNOWLEDGE = {
    "overwatch": """你是《守望先锋2》(Overwatch 2) 集锦审片员，精通其 HUD：
- 击杀播报(kill feed)位于画面**右上角**：击杀者名 + 武器图标 + 被击杀者名，短时间多行即连杀；
- 画面**中央偏下**的大字播报为团队事件：TEAM KILL(团灭)、英雄名+大招连招提示等；
- 屏幕下方中央是血条/技能冷却；左下角是队友状态；顶部中央是比分/时间；
- 死亡回放(death replay)与观战视角没有第一人称 HUD，注意区分。""",
    "csgo2": """你是《CS2》(Counter-Strike 2) 集锦审片员，精通其 HUD：
- 击杀播报(kill feed)位于画面**右上角**：击杀者名[武器]被击杀者名，爆头有特殊标记；
- **只有带红色边框/红色高亮的播报行才是"当前视角玩家"的击杀**——kill_count 只数带红框的行；同一玩家连续 2 行红框即双杀、3 行即三杀（含底部中央的击杀卡牌，卡片数字即本回合连杀数）；
- 没有红框的播报行是**别人的击杀**，不计入 kill_count（即使连续多行）；
- 画面中上方的**横幅播报**：回合胜利/失败、五杀 ACE、残局胜利等大事件；
- 顶部中央是回合数与比分；底部左侧是血量/护甲/金钱；右下角是武器与弹药；
- 死亡后进入观战视角（跟随队友第一人称）：底部出现「下一名玩家」提示条、信息卡显示被观战者名字——此时红框行是被观战队友的击杀，不是录像玩家的，注意区分；
- 雷达在左上角；残局(clutch)的标志是队友全灭后播报"X 对 N"的对抗提示。""",
    "valorant": """你是《瓦罗兰特》(VALORANT) 集锦审片员，精通其 HUD：
- 击杀播报(kill feed)位于画面**右侧中部**：击杀者名+武器图标+被击杀者名；**kill_count 的判定方法**：数 kill feed 中"同一击杀者名"短时间连续出现的行数（连杀还会弹出彩色横幅：双重击杀/三重击杀/四重击杀/五杀ACE）；
- kill feed 也会显示队友的击杀，但**同一玩家连续多行 = 连杀**，无论敌我都是精彩瞬间，务必计入 kill_count；
- 画面**中央上部的彩色大横幅**：连杀播报、五杀 ACE；
- 顶部中央是回合比分与时间；底部左侧是血量/护甲；底部中央是技能（含大招充能）；
- 死亡后进入观战视角，注意区分；
- 爆能器(spike)相关：下包/拆弹有全局播报与计时图标。""",
}


def build_system_prompt(game: str) -> str:
    """按游戏拼装 VLM system prompt（HUD 知识 + 事件枚举 + 评分口径）。"""
    game = game if game in GAME_KNOWLEDGE else "overwatch"
    names = event_names(game)
    etypes = "|".join(k for k in names if k != "not_game") + "|not_game"
    scoring = {
        "overwatch": "评分口径：团灭/三杀以上 8-10；双杀或关键大招 6-8；普通单杀 3-5；平淡画面或非对局 0-2。",
        "csgo2": "评分口径：五杀ACE或关键残局 8-10；三杀以上/关键拆弹 6-8；普通单杀或双杀 3-5；平淡画面或非对局 0-2。",
        "valorant": "评分口径：五杀ACE或关键残局 8-10；三杀以上/关键大招或拆弹 6-8；普通单杀或双杀 3-5；平淡画面或非对局 0-2。",
    }[game]
    return f"""{GAME_KNOWLEDGE[game]}
你将看到同一段录像按时间顺序排列的若干关键帧。请判断这段画面里发生了什么，输出**严格 JSON**（不要多余文字）：
{{"is_highlight": bool, "event_type": "{etypes}",
 "kill_count": int, "excitement": 0到10的整数, "confidence": 0到1的小数, "commentary": "不超过20字的中文解说"}}
{scoring}
kill_count 不确定就填 0。confidence 表示你对判断的把握。"""


class ReviewNotReady(ToolError):
    """The review layer cannot run (no key / bad config)."""


# ---------------------------------------------------------------- keyframes
def _pick_times(seg: dict, n: int) -> list[float]:
    """Pick up to n representative timestamps inside a segment.

    Event times are the moments a HUD announcement was actually visible.
    Windows are cut to maximize event density, so events cluster at the head
    of the segment - sampling them evenly can skip the exact frame where a
    kill feed is fully on screen (e.g. the 3rd kill landing). Cover the head
    events first, plus start/end bookends.
    """
    start, end = float(seg["start"]), float(seg["end"])
    events = sorted(set(float(t) for t in seg.get("event_times", [])))
    picks = {min(end - 0.2, start + 0.5), max(start + 0.2, end - 0.5)}
    if events:
        picks.update(events[:max(1, n - 1)])
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


def _norm_review(obj: dict, game: str = "overwatch") -> dict:
    """Validate & clamp the model's JSON into a predictable shape."""
    table = _game_table(game)
    et = str(obj.get("event_type", "other")).strip().lower()
    if et not in table:
        # 跨游戏词兜底：模型偶尔输出其它游戏的事件词（如 CS2 场景报 team_wipe）
        et = _CROSS_GAME_ALIAS.get(et, "other")
        if et not in table:
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
    # kill_count 对事件类型的升级规则（按游戏）
    if game in ("csgo2", "valorant"):
        if kills >= 5 and et in ("single_kill", "multi_kill", "other"):
            et, is_hl = "ace", True
        elif kills >= 2 and et in ("single_kill", "other"):
            et = "multi_kill"
    else:
        if kills >= 3 and et in ("single_kill", "other"):
            et, is_hl = "multi_kill", True
        elif kills == 2 and et in ("single_kill", "other"):
            et = "multi_kill"
    if et in ("team_wipe", "ace", "clutch"):
        is_hl = True
    comment = str(obj.get("commentary") or "").strip().replace("\n", " ")[:40]
    return {"is_highlight": is_hl, "event_type": et, "kill_count": kills,
            "excitement": exc, "confidence": round(conf, 2),
            "event_label": table[et][0], "commentary": comment}


def _l1_hint(seg: dict) -> str:
    """Tell the VLM when the local detector saw kill-feed activity.

    L1 already knows *when* HUD announcements appeared - handing those
    timestamps to the VLM focuses its attention on the right frames and
    dramatically improves kill counting (L1 -> L2 information hand-off).
    """
    kt = seg.get("kill_times") or []
    if not kt:
        return ""
    shown = ", ".join(f"{t:.1f}s" for t in kt[:8])
    more = f" 等 {len(kt)} 次" if len(kt) > 8 else ""
    return (f"本地信号检测器在这段捕获到 {len(kt)} 次播报区画面变化（{shown}{more}），"
            f"变化可能来自击杀播报，也可能来自其他 HUD 元素，请以关键帧实际内容为准。")


def _ocr_hint(seg: dict) -> str:
    """Feed the OCR-read kill feed text to the VLM as ground truth.

    OCR reads the kill feed frame by frame - it knows exactly who killed
    whom and how many in a row. Handing that to the VLM as factual context
    fixes its chronic mis-counting (it used to call triple kills single).
    An explicit "no kills seen" statement is just as valuable: it stops the
    model from hallucinating kills out of L1's looser HUD-change hints.
    """
    if "ocr_kills" not in seg:
        return ""  # OCR 没跑过这段：不给任何信息
    ks = seg.get("ocr_kills") or []
    if not ks:
        return ("OCR 逐帧读取了该段的击杀播报区，未发现任何击杀条目——"
                "kill_count 应为 0，除非关键帧里有 OCR 没覆盖到的明确击杀证据。")
    parts = [f"{k['killer']}击杀{k['victim']}" for k in ks[:8]]
    more = f" 等 {len(ks)} 条" if len(ks) > 8 else ""
    max_streak = max(k["streak"] for k in ks)
    text = (f"OCR 逐帧读取的击杀播报（确定性事实，以此为准）：{'；'.join(parts)}{more}。")
    if max_streak >= 2:
        top = max(ks, key=lambda k: k["streak"])
        text += f"其中 {top['killer']} 连杀 {max_streak} 人，kill_count 不得低于 {max_streak}。"
    return text


def _card_hint(seg: dict) -> str:
    """Feed the CS2 kill-card count to the VLM as ground truth.

    The card stack above the rank badge only counts the RECORDER's kills
    this round - unlike the feed (everyone's kills). When the card channel
    confirmed kills in this segment, kill_count can not go below that.
    """
    ks = seg.get("card_kills") or []
    if not ks:
        return ""
    total = sum(k["kills"] for k in ks)
    top = max(k["streak"] for k in ks)
    return (f"击杀卡牌（HUD 本回合计数，只计录像玩家本人的击杀）确认玩家在此段"
            f"落地 {total} 杀（本回合累计第 {top} 杀），kill_count 不得低于 {total}。")


def _player_hint(player: str) -> str:
    """Name the recorder for the VLM so commentary doesn't rename them.

    The VLM half-reads player names from keyframes and invents spellings
    ("DAnTe" -> "Jaden"). When the OCR channel has anchored the recorder's
    actual feed name, stating it explicitly keeps the model on-script and
    tells it which feed rows are the player's own kills.
    """
    if not player:
        return ""
    return (f"录像玩家（第一人称视角本人）是「{player}」，"
            f"kill feed 中该名字的行就是玩家本人的击杀。")


def _apply_ocr_override(seg: dict, review: dict, game: str) -> None:
    """OCR counted the feed rows deterministically - trust it over the VLM.

    If OCR saw a streak the VLM under-counted, correct kill_count (and the
    event type it upgrades to). VLM keeps the final say on excitement -
    that is a judgement about the whole scene, not a count.
    """
    ks = seg.get("ocr_kills") or []
    floor = max((k["streak"] for k in ks), default=0)
    # 击杀卡牌/骷髅头确认只计玩家本人的击杀：是 kill_count 更硬的下限
    floor = max(floor, sum(k["kills"] for k in (seg.get("card_kills") or [])))
    floor = max(floor, int(seg.get("confirmed_kills") or 0))
    if floor < 2 or review.get("kill_count", 0) >= floor:
        return
    review["kill_count"] = floor
    review["ocr_override"] = True
    et = review.get("event_type", "other")
    if game in ("csgo2", "valorant"):
        if floor >= 5:
            review["event_type"] = "ace"
        elif et in ("single_kill", "other"):
            review["event_type"] = "multi_kill"
    elif floor >= 3 and et in ("single_kill", "other"):
        review["event_type"] = "multi_kill"
    review["event_label"] = event_names(game).get(review["event_type"], review["event_label"])
    if review["event_type"] in ("multi_kill", "ace", "team_wipe"):
        review["is_highlight"] = True


def _call_vlm(video: Path, seg: dict, cfg: dict, key: str,
              player: str = "") -> dict:
    """One API call: N keyframes of one segment -> normalized review dict."""
    rv = cfg["review"]
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    frames = []
    for t in _pick_times(seg, int(rv.get("frames_per_segment", 4))):
        frames.append((t, _extract_frame(video, t, int(rv.get("frame_width", 768)))))
    if not frames:
        raise ToolError("没有可用的关键帧")

    content = [{
        "type": "text",
        "text": (f"片段 #{seg.get('rank', seg.get('id', '?'))}，"
                 f"源片 {seg['start']:.1f}s - {seg['end']:.1f}s。"
                 + _player_hint(player)
                 + _ocr_hint(seg)
                 + _card_hint(seg)
                 + _l1_hint(seg)
                 + f"下面 {len(frames)} 张是按时间顺序的关键帧，判断后输出 JSON。"),
    }]
    for _, jpg in frames:
        b64 = base64.b64encode(jpg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": rv.get("model", "deepseek-flash"),
        "messages": [{"role": "system", "content": build_system_prompt(game)},
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

    obj: dict = {}
    text = ""
    # 多图 + 低推理档位偶发输出非 JSON（被截断/夹带推理文本），空结果重试一次
    for attempt in range(2):
        with urllib.request.urlopen(req, timeout=float(rv.get("timeout_sec", 180))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage", {})
        msg = (data.get("choices") or [{}])[0].get("message", {})
        text = msg.get("content") or ""
        if not text:
            # 推理模型可能把 token 预算耗在 reasoning 上导致 content 为空，
            # 从 reasoning_content 兜底提取（模型推理中通常已写出答案）
            reasoning = msg.get("reasoning_content") or ""
            if reasoning:
                obj = _extract_json(reasoning)
                if obj:
                    log("  content 为空，已从 reasoning_content 提取结果", "warn")
                    break
        obj = _extract_json(text)
        if obj:
            break
        if attempt == 0:
            log(f"  模型输出无法解析为 JSON（{len(text)} 字符），重试一次", "warn")
    out = _norm_review(obj, game)
    out["tokens"] = usage.get("total_tokens", 0)
    out["frames"] = len(frames)
    return out


# ---------------------------------------------------------------- pipeline
def _blended_score(seg: dict, review: dict, rv: dict, game: str) -> float:
    """Fuse VLM semantics with the signal score:  w_s*signal + w_v*(exc/10) + w_e*event."""
    ws = float(rv.get("w_signal", 0.35))
    wv = float(rv.get("w_vlm", 0.45))
    we = float(rv.get("w_event", 0.20))
    ev = event_weights(game).get(review["event_type"], 0.4)
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
    game = cfg.get("recognizer", {}).get("game", "overwatch")
    player = str(tl.get("player") or "")
    max_seg = int(rv.get("max_segments", 8))
    todo = selected[:max_seg]
    log(f"语义复核（{GAME_NAMES.get(game, game)}）：{len(todo)}/{len(selected)} 段，"
        f"每段最多 {rv.get('frames_per_segment', 4)} 关键帧，"
        f"模型 {rv.get('model', 'deepseek-flash')}（{rv.get('provider', 'deepseek')} 云端）"
        + (f"，玩家「{player}」" if player else ""), "review")

    def work(seg):
        # 多素材：段自带源文件（时间已是素材内局部时间），单素材回退全局 video
        vid = Path(seg.get("src_video") or video)
        return seg, _call_vlm(vid, seg, cfg, key, player=player)

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
        _apply_ocr_override(seg, review, game)
        seg["score_signal"] = seg["score"]
        seg["score"] = _blended_score(seg, review, rv, game)
        fixed = "（OCR 修正连杀）" if review.get("ocr_override") else ""
        seg["reason"] = (f"{review['event_label']}"
                         + (f"×{review['kill_count']}" if review["kill_count"] else "")
                         + f"、VLM精彩 {review['excitement']}/10{fixed}"
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
        "game": game,
        "reviewed": len(results), "failed": len(errors),
        "dropped": len(dropped), "total_tokens": total_tokens,
    }
    log(f"复核完成：成功 {len(results)} / 失败 {len(errors)} / 丢弃 {len(dropped)} 段，"
        f"消耗约 {total_tokens} tokens", "review")
    return tl["review_summary"]
