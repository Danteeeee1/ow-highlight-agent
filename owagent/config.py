"""Configuration loading, validation and CLI override merging."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "roi": {
        "x": 0.72, "y": 0.05, "w": 0.27, "h": 0.22
    },
    "sample": {"fps": 8, "width": 96, "height": 32},
    "recognizer": {
        "_comment": "P1 神经网络识别后端：none=启发式（默认）；crispy=Crispy 预训练击杀检测",
        "backend": "none",
        "game": "overwatch",
        "weights_dir": "models/crispy",
        "confidence": 0.6,
        "fps": 8,
        "stretch": False
    },
    "review": {
        "_comment": "L2 语义复核：VLM 看入选片段关键帧，判定事件类型/精彩度并写解说（DeepSeek 云端）",
        "enabled": False,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "OWAGENT_REVIEW_KEY",
        "api_key_file": "config.local.json",
        "frames_per_segment": 4,
        "max_segments": 8,
        "frame_width": 768,
        "effort": "low",
        "max_tokens": 3000,
        "timeout_sec": 180,
        "concurrency": 2,
        "drop_non_highlight": True,
        "w_signal": 0.35,
        "w_vlm": 0.45,
        "w_event": 0.20
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
    "select": {"top_k": 6, "max_total_sec": 90.0, "min_score": 0.0},
    "render": {
        "width": 1280, "height": 720, "fps": 30, "crf": 23, "preset": "veryfast",
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
    rec = cfg["recognizer"]
    if rec["backend"] not in ("none", "crispy"):
        raise ValueError(f"recognizer.backend 只能是 none / crispy，当前为 {rec['backend']}")
    if rec["game"] not in ("overwatch", "csgo2", "valorant"):
        raise ValueError(f"recognizer.game 只能是 overwatch / csgo2 / valorant，当前为 {rec['game']}")
    if not 0.05 <= float(rec["confidence"]) <= 0.99:
        raise ValueError("recognizer.confidence 建议在 0.4~0.8 之间")
    if float(rec["fps"]) <= 0:
        raise ValueError("recognizer.fps 必须为正数")
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
    if getattr(args, "game", None):
        c["recognizer"]["game"] = args.game
    if getattr(args, "confidence", None) is not None:
        c["recognizer"]["confidence"] = float(args.confidence)
    if getattr(args, "review", False):
        c["review"]["enabled"] = True
    if getattr(args, "no_review", False):
        c["review"]["enabled"] = False
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
