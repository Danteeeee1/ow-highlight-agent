"""Configuration loading, validation and CLI override merging."""
from __future__ import annotations

import copy
import json
from pathlib import Path

# 各游戏启发式检测的默认 ROI（比例坐标，覆盖对应游戏的击杀播报区）。
# 仅在显式指定 --game 且用户没有手动给 --roi / config 覆盖时套用；
# crispy 后端不使用此 ROI（模型自带 HUD 框）。
GAME_ROI = {
    # OW2 击杀播报：右上角
    "overwatch": {"x": 0.72, "y": 0.05, "w": 0.27, "h": 0.22},
    # CS2 击杀播报：右上角。y 从 0.06 起——上面的网络统计条(延迟/丢包/FPS)
    # 每秒跳字，会让墨迹上升启发式产生大量假事件
    "csgo2": {"x": 0.70, "y": 0.06, "w": 0.29, "h": 0.16},
    # VALORANT 击杀播报：右侧中部（约屏幕 1/4~2/3 高度处）
    "valorant": {"x": 0.73, "y": 0.24, "w": 0.26, "h": 0.32},
}

DEFAULT_CONFIG = {
    "roi": dict(GAME_ROI["overwatch"]),
    "sample": {"fps": 8, "width": 96, "height": 32},
    "recognizer": {
        "_comment": "P1 神经网络识别后端：none=启发式（默认）；crispy=Crispy 预训练击杀检测",
        "backend": "none",
        "game": "overwatch",
        "weights_dir": "models/crispy",
        "confidence": 0.6,
        "fps": 8,
        "stretch": "auto"
    },
    "review": {
        "_comment": "L2 语义复核：VLM 看入选片段关键帧，判定事件类型/精彩度并写解说（DeepSeek 云端）",
        "enabled": False,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "OWAGENT_REVIEW_KEY",
        "api_key_file": "config.local.json",
        "frames_per_segment": 6,
        "max_segments": 8,
        "frame_width": 768,
        "effort": "low",
        "max_tokens": 8000,
        "timeout_sec": 180,
        "concurrency": 2,
        "drop_non_highlight": True,
        "w_signal": 0.35,
        "w_vlm": 0.45,
        "w_event": 0.20
    },
    "ocr": {
        "_comment": "L1 OCR 通道：读击杀播报文字拿到击杀者/被杀者/连杀数（rapidocr，离线）。"
                    "mode=annotate 只识别入选片段（默认，快）；mode=full 全片独立召回（慢）。"
                    "需要 pip install -r requirements-ocr.txt；缺失时自动跳过",
        "enabled": True,
        "mode": "annotate",
        "fps": 1.0,
        "min_score": 0.5,
        "still_diff": 2.5,
        "streak_window": 6.0,
        "feed_persist_sec": 25.0,
        "player_name": "",
        "max_side_len": 320
    },
    "skull": {
        "_comment": "OW 专属通道：屏幕中心击杀确认骷髅头（只有击杀者本人能看到，"
                    "是最专属的玩家击杀信号）。眼窝双孔签名 + 生存期过滤；"
                    "confirm_feed 用击杀播报确认并附带击杀者/被击杀者名字。"
                    "仅 overwatch 生效；依赖 opencv（requirements-ocr.txt），缺失自动跳过",
        "enabled": True,
        "fps": 16,
        "confirm_feed": True
    },
    "killcard": {
        "_comment": "CS2 专属通道：底部中央回合击杀卡牌堆（每杀一张卡，顶卡数字=本回合"
                    "击杀数，只计录像玩家自己的击杀）。不连续扫描：只在其他召回通道"
                    "的事件时刻前后读卡牌数字做差分（新卡弹出动画期读数置信度 0.97+）。"
                    "仅 csgo2 生效；ROI 在 1280x960 4:3 实测，16:9 未验证可调；"
                    "依赖 rapidocr（requirements-ocr.txt），缺失自动跳过",
        "enabled": True,
        "roi": {"x": 0.455, "y": 0.862, "w": 0.135, "h": 0.098}
    },
    "redbox": {
        "_comment": "CS2 专属通道：右上击杀播报红框（当前视角玩家的击杀行整行红色"
                    "高亮/描边，非击杀者名字颜色）。与击杀卡牌互证：红框+卡牌变化="
                    "视角玩家的击杀，缺一则不确认。只在召回事件时刻前后读帧，纯像素"
                    "无 OCR。仅 csgo2 生效；依赖 opencv，缺失自动跳过",
        "enabled": True
    },
    "pov": {
        "_comment": "CS2 专属通道：视角判定。死亡观战/GOTV 时底部出现『下一名玩家』"
                    "控制条、信息卡显示被观战者名字——此时红框/卡牌变化属于被观战者"
                    "而非录像主人。观战他人区间内的击杀确认会被撤销，整段从成片剔除。"
                    "目标用户名取 ocr.player_name，缺省用本人视角锚点的播报击杀者投票。",
        "enabled": True
    },
    "nameplate": {
        "_comment": "OW 专属通道：左下角玩家名牌。名牌是唯一持续印着录像者名字的"
                    "HUD——读到的名字注入 ocr.player_name（用户已配置时不覆盖），"
                    "并与右上播报行协同：击杀者名=名牌名 → 本人击杀铁证；骷髅头确认"
                    "事件的 killer 名与名牌名不符则撤销确认。观战他人时名牌换成被观战"
                    "者名，连续非主名段整段剔除。仅 overwatch 生效；"
                    "ROI 按 1920x1080 16:9 实测，依赖 rapidocr，缺失自动跳过",
        "enabled": True,
        "roi": {"x": 0.031, "y": 0.852, "w": 0.26, "h": 0.13},
        "fps": 0.5
    },
    "detect": {
        "diff_gain": 60.0,
        "rise_gain": 0.25,
        "ink_gain": 0.45,
        "w_diff": 0.45, "w_rise": 0.40, "w_ink": 0.15,
        "threshold": 0.55,
        "min_rise": 0.08,
        "refractory_sec": 0.6
    },
    "audio": {
        "enabled": True, "sample_rate": 8000, "window_sec": 0.10,
        "peak_ratio": 2.6, "refractory_sec": 0.5, "baseline_alpha": 0.05,
        "min_rms": 0.012
    },
    "clip": {
        "pre_roll": 4.0, "post_roll": 3.0, "merge_gap": 5.0,
        "min_len": 3.0, "max_len": 16.0
    },
    "select": {"top_k": 6, "max_total_sec": 120.0, "min_score": 0.0,
               "playerless_penalty": 0.15},
    "render": {
        "_comment": "width/height/fps 为 0 表示跟随源（推荐）：成片保持录像原始分辨率/帧率，"
                    "不做降采样。crf 18 接近视觉无损；烧字幕时 parts 走近无损中间档，"
                    "最终画质只由烧录阶段编码一次决定",
        "width": 0, "height": 0, "fps": 0, "crf": 18, "preset": "medium",
        "vertical": False, "burn_subtitles": False, "bgm": None, "bgm_volume": 0.18
    },
    "agent": {
        "llm_enabled": False,
        "api_key_env": "OWAGENT_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "timeout_sec": 15
    }
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with p.open("r", encoding="utf-8") as f:
            cfg = _merge(cfg, json.load(f))
    return cfg


def validate(cfg: dict, meta: dict | None = None) -> None:
    roi = cfg["roi"]
    for k in ("x", "y", "w", "h"):
        if not 0 <= roi[k] <= 1:
            raise ValueError(f"roi.{k} 必须在 0~1 之间，当前为 {roi[k]}")
    if roi["x"] + roi["w"] > 1.001 or roi["y"] + roi["h"] > 1.001:
        raise ValueError("ROI 超出了画面范围 (x+w / y+h 不能大于 1)")
    if cfg["sample"]["fps"] <= 0 or cfg["sample"]["width"] <= 0:
        raise ValueError("sample.fps / width / height 必须为正数")
    if not 0 < cfg["detect"]["threshold"] < 1.5:
        raise ValueError("detect.threshold 建议在 0.2~1.0 之间")
    if cfg["clip"]["pre_roll"] < 0 or cfg["clip"]["post_roll"] < 0:
        raise ValueError("pre_roll / post_roll 不能为负")
    if cfg["select"]["top_k"] < 1:
        raise ValueError("select.top_k 至少为 1")
    if not 0.0 <= float(cfg["select"].get("playerless_penalty", 0.15)) <= 0.5:
        raise ValueError("select.playerless_penalty 建议在 0~0.3 之间（0=关闭玩家中心性降权）")
    rec = cfg["recognizer"]
    if rec["backend"] not in ("none", "crispy"):
        raise ValueError(f"recognizer.backend 只能是 none / crispy，当前为 {rec['backend']}")
    if rec["game"] not in ("overwatch", "csgo2", "valorant"):
        raise ValueError(f"recognizer.game 只能是 overwatch / csgo2 / valorant，当前为 {rec['game']}")
    if rec.get("stretch") not in (True, False, "auto"):
        raise ValueError('recognizer.stretch 只能是 true / false / "auto"')
    if not 0.05 <= float(rec["confidence"]) <= 0.99:
        raise ValueError("recognizer.confidence 建议在 0.4~0.8 之间")
    if float(rec["fps"]) <= 0:
        raise ValueError("recognizer.fps 必须为正数")
    oc = cfg.get("ocr", {})
    if oc.get("mode", "annotate") not in ("annotate", "full"):
        raise ValueError('ocr.mode 只能是 "annotate" / "full"')
    if float(oc.get("fps", 1.0)) <= 0:
        raise ValueError("ocr.fps 必须为正数")
    if not 0.1 <= float(oc.get("min_score", 0.5)) <= 0.99:
        raise ValueError("ocr.min_score 建议在 0.3~0.8 之间")
    if float(oc.get("streak_window", 6.0)) <= 0:
        raise ValueError("ocr.streak_window 必须为正数")
    sk = cfg.get("skull", {})
    if float(sk.get("fps", 16)) <= 0:
        raise ValueError("skull.fps 必须为正数")
    rd = cfg.get("render", {})
    for k in ("width", "height", "fps"):
        if int(rd.get(k, 0)) < 0:
            raise ValueError(f"render.{k} 不能为负（0 = 跟随源）")
    if not 0 <= int(rd.get("crf", 18)) <= 51:
        raise ValueError("render.crf 必须在 0~51 之间（18 接近视觉无损）")
    if str(rd.get("preset", "medium")) not in (
            "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
            "slow", "slower", "veryslow"):
        raise ValueError(f"render.preset 不是有效的 x264 预设: {rd.get('preset')}")
    rv = cfg.get("review", {})
    if rv.get("enabled"):
        if rv.get("provider") != "deepseek":
            raise ValueError(f"review.provider 目前仅支持 deepseek，当前为 {rv.get('provider')}")
        if not str(rv.get("model", "")).strip():
            raise ValueError("review.model 不能为空")
        if not 0.05 <= float(rv.get("w_signal", 0.35)) <= 1:
            raise ValueError("review.w_signal 建议在 0.2~0.5 之间")
        if int(rv.get("frames_per_segment", 4)) < 1:
            raise ValueError("review.frames_per_segment 至少为 1")


def apply_overrides(cfg: dict, args) -> dict:
    """Apply argparse overrides without mutating the caller's dict."""
    c = copy.deepcopy(cfg)
    if getattr(args, "roi", None):
        try:
            x, y, w, h = [float(v) for v in args.roi.split(",")]
        except ValueError:
            raise ValueError("--roi 需要 4 个数字，例如 0.72,0.05,0.27,0.22")
        c["roi"].update({"x": x, "y": y, "w": w, "h": h})
    simple = {
        "fps": ("sample", "fps"),
        "threshold": ("detect", "threshold"),
        "top_k": ("select", "top_k"),
        "pre": ("clip", "pre_roll"),
        "post": ("clip", "post_roll"),
        "gap": ("clip", "merge_gap"),
    }
    for attr, (section, key) in simple.items():
        val = getattr(args, attr, None)
        if val is not None:
            c[section][key] = val
    if getattr(args, "no_audio", False):
        c["audio"]["enabled"] = False
    if getattr(args, "vertical", False):
        c["render"]["vertical"] = True
    if getattr(args, "burn", False):
        c["render"]["burn_subtitles"] = True
    if getattr(args, "bgm", None):
        c["render"]["bgm"] = str(Path(args.bgm).resolve())
    if getattr(args, "llm", False):
        c["agent"]["llm_enabled"] = True
    if getattr(args, "recognizer", None):
        c["recognizer"]["backend"] = args.recognizer
    game_arg = getattr(args, "game", None)
    if game_arg:
        c["recognizer"]["game"] = game_arg
        # 切换到非默认游戏且未手动指定 ROI 时，套用该游戏的启发式默认 ROI。
        # overwatch 不动 config 里已有的 roi（保持老用户自定义不受影响）；
        # crispy 后端不读 roi，此切换只影响启发式路径与 ROI 预览。
        if game_arg != "overwatch" and game_arg in GAME_ROI and not getattr(args, "roi", None):
            c["roi"] = dict(GAME_ROI[game_arg])
    if getattr(args, "confidence", None) is not None:
        c["recognizer"]["confidence"] = float(args.confidence)
    if getattr(args, "review", False):
        c["review"]["enabled"] = True
    if getattr(args, "no_review", False):
        c["review"]["enabled"] = False
    if getattr(args, "ocr", False):
        c["ocr"]["enabled"] = True
    if getattr(args, "no_ocr", False):
        c["ocr"]["enabled"] = False
    if getattr(args, "player", None):
        c["ocr"]["player_name"] = str(args.player).strip()
    validate(c)
    return c


def pixel_roi(cfg: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert fractional ROI to even, in-bounds pixel coordinates."""
    from .util import even
    r = cfg["roi"]
    x = even(int(r["x"] * width))
    y = even(int(r["y"] * height))
    w = even(min(int(r["w"] * width), width - x))
    h = even(min(int(r["h"] * height), height - y))
    return x, y, w, h
