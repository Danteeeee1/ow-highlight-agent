"""Learned recognizer backend (P1): the Crispy pre-trained kill detector.

This module integrates the neural network shipped by the MIT-licensed Crispy
project (https://github.com/Flowtter/crispy, release 1.1.4). It is a small
numpy MLP trained on real gameplay frames to answer one question per frame:
"is a kill announcement visible right now?" - which is exactly the recall
task our heuristic detector approximates with pixel statistics.

How it maps onto this project:
  * ffmpeg decodes a fixed HUD box at 8fps, after scaling the frame to the
    canonical resolution the network was trained on (1920x1080, 16:9).
  * The per-game preprocessing is replicated faithfully from Crispy's
    `api/models/highlight.py` (red-channel boost for OW2, edge detection +
    mask overlay for CS2 / VALORANT).
  * Every frame gets a kill probability; a rising edge over `confidence`
    becomes one model event with the same schema as heuristic events, so the
    downstream scoring / clipping pipeline is unchanged.

Dependencies: numpy (always) and Pillow (CS2 / VALORANT preprocessing only).
Both live in requirements-model.txt and are optional: without them the
pipeline simply falls back to the heuristic detector.

Weights and masks are NOT committed to git - run `python run.py fetch-model`
to download them (about 21 MB, one-time) into models/crispy/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from .util import ToolError, ffmpeg_bin, log, read_exact

ROOT = Path(__file__).resolve().parent.parent

# One-time download source (public release asset, no auth needed).
CRISPY_RELEASE_API = "https://api.github.com/repos/Flowtter/crispy/releases/tags/1.1.4"

# Canonical HUD box per game, replicated from Crispy's Box(...) calls:
#   (x_offset_from_center, y, width, height, shift_x)
# x is measured from the center of the canonical frame (half - x + shift_x).
CRISPY_BOXES = {
    "overwatch": (50, 490, 100, 100, 0),
    "csgo2": (50, 925, 100, 100, 20),
    "valorant": (61, 801, 122, 62, 0),
}


class ModelNotReady(ToolError):
    """The learned recognizer cannot run (missing weights / deps)."""


def weights_dir(cfg: dict) -> Path:
    d = Path(cfg["recognizer"].get("weights_dir") or (ROOT / "models" / "crispy"))
    if not d.is_absolute():
        d = ROOT / d
    return d


def _expit(z):
    import numpy as np
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


class CrispyNetwork:
    """The 10000-120-15-2 MLP, queried directly with numpy (no DL framework)."""

    def __init__(self, game: str, weights_path: Path):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ModelNotReady(
                f"识别器需要 numpy（{exc}）。安装：pip install -r requirements-model.txt") from exc
        self.np = np
        try:
            self.layers = [np.asarray(w, dtype=float) for w in
                           np.load(weights_path, allow_pickle=True)]
        except FileNotFoundError as exc:
            raise ModelNotReady(
                f"找不到权重文件 {weights_path.name}。"
                f"先执行: python run.py fetch-model --game {game}") from exc
        expected = {10000: 3, 4000: 3}[self.layers[0].shape[1]]
        if len(self.layers) != expected:
            raise ModelNotReady(f"{weights_path.name} 结构异常: "
                                f"{[w.shape for w in self.layers]}")

    def query(self, inputs) -> list[float]:
        x = inputs
        for w in self.layers:
            x = _expit(w @ x)
        return x


# --------------------------------------------------------------------------
# per-game preprocessing (faithful ports of Crispy's highlight.py)
# --------------------------------------------------------------------------

def _preprocess_overwatch(frame, np):
    """Crispy OW2 pipeline, vectorized: red-channel boost -> grayscale."""
    img = np.frombuffer(frame, dtype=np.uint8).reshape(100, 100, 3)
    r = img[:, :, 0].astype(np.int64)
    g = img[:, :, 1].astype(np.int64)
    b = img[:, :, 2].astype(np.int64)

    strong = (r > 200) & (g < 100) & (b < 100)
    mid = (r > 200) & (g < 180) & (b < 180) & ~strong

    # branch values are computed from the ORIGINAL channels (as in the
    # per-pixel loop), then merged
    r2 = np.where(strong, 255, np.where(mid, np.minimum(np.floor(r * 1.5), 255),
                                        np.minimum(np.floor(r * 1.1), 255)))
    g2 = np.where(strong, 255, np.where(mid, np.maximum(np.floor(g * 0.3), 0),
                                        np.maximum(np.floor(g * 0.05), 0)))
    b2 = np.where(strong, 255, np.where(mid, np.maximum(np.floor(b * 0.3), 0),
                                        np.maximum(np.floor(b * 0.05), 0)))
    # PIL's RGB->L fixed point conversion (19595/38470/7471, rounded)
    total = (r2 * 19595 + g2 * 38470 + b2 * 7471 + 32768).astype(np.int64)
    return (total >> 16).astype(np.uint8)


def _preprocess_csgo2(frame, np, mask):
    from PIL import Image, ImageFilter, ImageOps
    im = Image.frombytes("RGB", (100, 100), frame)
    im = ImageOps.grayscale(im.filter(ImageFilter.FIND_EDGES)
                            .filter(ImageFilter.EDGE_ENHANCE_MORE))
    final = Image.new("RGB", (100, 100))
    final.paste(im, (0, 0))
    final.paste(mask, (0, 0), mask)
    return np.asarray(final.getchannel("R"), dtype=np.uint8)


def _preprocess_valorant(frame, np, mask):
    from PIL import Image, ImageFilter, ImageOps

    def apply_filter(image, image_filter):
        image = image.filter(image_filter)
        image = image.crop((1, 1, image.width - 2, image.height - 2))
        image.paste(mask, (0, 0), mask)
        left = image.crop((0, 0, 25, 60))
        right = image.crop((95, 0, 120, 60))
        final = Image.new("RGB", (50, 60))
        final.paste(left, (0, 0))
        final.paste(right, (25, 0))
        return final.crop((0, 20, 50, 60))

    im = Image.frombytes("RGB", (122, 62), frame)
    im = ImageOps.grayscale(im)
    edges = apply_filter(im, ImageFilter.FIND_EDGES)
    enhanced = apply_filter(im, ImageFilter.EDGE_ENHANCE_MORE)
    final = Image.new("RGB", (50, 80))
    final.paste(edges, (0, 0))
    final.paste(enhanced.transpose(Image.FLIP_TOP_BOTTOM), (0, 40))
    return np.asarray(final.getchannel("R"), dtype=np.uint8)


_PREPROCESS = {"overwatch": _preprocess_overwatch,
               "csgo2": _preprocess_csgo2,
               "valorant": _preprocess_valorant}


def box_pixels(game: str, stretch: bool) -> tuple[int, int, int, int]:
    """HUD box in the canonical (scaled) frame, mirroring Crispy's Box()."""
    x_off, y, w, h, shift = CRISPY_BOXES[game]
    half = 720 if stretch else 960
    return half - x_off + shift, y, w, h


def canonical_size(stretch: bool) -> tuple[int, int]:
    return (1440, 1080) if stretch else (1920, 1080)


# --------------------------------------------------------------------------
# main entry: model-based event recall
# --------------------------------------------------------------------------

def detect_model(video: str | Path, cfg: dict, meta: dict) -> dict:
    """Same contract as detector.detect_visual, but events come from the NN."""
    rec = cfg["recognizer"]
    game = rec["game"]
    if game not in CRISPY_BOXES:
        raise ModelNotReady(f"不支持的游戏 {game}，可选: {', '.join(CRISPY_BOXES)}")

    conf = float(rec["confidence"])
    fps = float(rec["fps"])
    stretch = bool(rec.get("stretch", False))
    w_dir = weights_dir(cfg)

    net = CrispyNetwork(game, w_dir / f"{game}.npy")
    np = net.np

    mask = None
    if game in ("csgo2", "valorant"):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ModelNotReady(
                f"{game} 预处理需要 Pillow（{exc}）。"
                "安装: pip install -r requirements-model.txt") from exc
        mask = Image.open(w_dir / f"{game}-mask.png").convert("RGBA")

    cw, ch = canonical_size(stretch)
    bx, by, bw, bh = box_pixels(game, stretch)
    frame_size = bw * bh * 3
    preprocess = _PREPROCESS[game]

    vf = f"fps={fps},scale={cw}:{ch},crop={bw}:{bh}:{bx}:{by}"
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    log(f"Crispy 模型识别 ({game})：归一 {cw}x{ch} → HUD 框 x={bx} y={by} "
        f"w={bw} h={bh} @ {fps}fps，confidence={conf}", "model")

    events: list[dict] = []
    signals: list[dict] = []
    prev_p = 0.0
    idx = 0

    with tempfile.TemporaryFile(mode="w+b") as errfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile)
        assert proc.stdout is not None
        try:
            while True:
                frame = read_exact(proc.stdout, frame_size)
                if len(frame) < frame_size:
                    break
                t = idx / fps
                gray = preprocess(frame, np, mask) if game != "overwatch" \
                    else preprocess(frame, np)
                inputs = (gray.astype(float).ravel() / 255.0 * 0.99) + 0.01
                p = float(net.query(inputs)[1])  # index 1 = P(kill)
                signals.append({"t": round(t, 3), "p_kill": round(p, 4)})
                # rising edge over confidence = one fresh kill announcement
                if p >= conf and prev_p < conf:
                    events.append({"t": round(t, 3), "source": "model",
                                   "strength": round(p, 4), "p_kill": round(p, 4)})
                prev_p = p
                idx += 1
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errfile.seek(0)
            detail = errfile.read().decode("utf-8", "replace").strip()[-1200:]
            log(f"模型解码失败: {detail}", "warn")

    log(f"模型扫描 {idx} 帧 / {idx / fps:.1f}s，召回 {len(events)} 个击杀事件 "
        f"(峰值 p={max((s['p_kill'] for s in signals), default=0):.2f})", "model")
    return {"events": events, "signals": signals, "frames": idx, "fps": fps,
            "backend": "crispy", "game": game, "confidence": conf,
            "box_px": [bx, by, bw, bh], "canonical": [cw, ch]}


# --------------------------------------------------------------------------
# weights download (one-time)
# --------------------------------------------------------------------------

def fetch_weights(games: list[str], dest: Path) -> list[str]:
    """Download the Crispy release once and pull out the needed .npy + masks."""
    dest.mkdir(parents=True, exist_ok=True)
    games = [g for g in games if g]
    if not games:
        raise ValueError("--game 至少填一个: overwatch / csgo2 / valorant / all")
    if "all" in games:
        games = list(CRISPY_BOXES)
    for g in games:
        if g not in CRISPY_BOXES:
            raise ValueError(f"未知游戏 {g}，可选: {', '.join(CRISPY_BOXES)} 或 all")

    wanted = {f"{g}.npy" for g in games}
    wanted |= {"csgo2-mask.png", "valorant-mask.png"}  # masks are tiny, always take
    existing = {f.name for f in dest.iterdir() if f.is_file()}
    if wanted <= existing:
        log("权重文件已齐全，无需下载", "model")
        return sorted(wanted)

    log(f"从 Crispy 官方 release 下载权重（约 21 MB，一次性）…", "model")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                CRISPY_RELEASE_API,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": "ow-highlight-agent"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                release = json.loads(resp.read().decode("utf-8"))
            asset = next((a for a in release.get("assets", [])
                         if a.get("name") == "windows.zip"), None)
            if not asset:
                raise ToolError("Crispy release 中找不到 windows.zip 资产")
            req = urllib.request.Request(
                f"https://api.github.com/repos/Flowtter/crispy/releases/assets/{asset['id']}",
                headers={"Accept": "application/octet-stream",
                         "User-Agent": "ow-highlight-agent"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            break
        except (OSError, ValueError, KeyError) as exc:
            if attempt == 2:
                raise ToolError(
                    f"下载失败: {exc}\n可手动下载 release 1.1.4 的 windows.zip，"
                    "把 assets/<游戏>.npy 与两个 *-mask.png 解压到 models/crispy/ 下。") from exc
            log(f"下载失败，重试 ({attempt + 2}/3)…", "warn")
            time.sleep(2)

    saved = []
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "windows.zip"
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                base = Path(name).name
                if base in wanted:
                    out = dest / base
                    with z.open(name) as src, open(out, "wb") as f:
                        shutil.copyfileobj(src, f)
                    saved.append(base)
                    log(f"已保存 {out}（{out.stat().st_size / 1024:.0f} KB）", "model")
    missing = wanted - {f.name for f in dest.iterdir()}
    if missing:
        raise ToolError(f"下载包中缺少文件: {', '.join(sorted(missing))}")
    return saved
