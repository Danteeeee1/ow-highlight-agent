"""Command line entry points for the OW Highlight Agent prototype."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .agent import orchestrate
from .audio import detect_audio
from .config import apply_overrides, load_config, pixel_roi
from .detector import detect_visual
from .killcard import KillcardNotReady, detect_killcard
from .model import CRISPY_BOXES, ModelNotReady, detect_model, fetch_weights
from .ocr import OCRNotReady, annotate_selected, detect_ocr
from .pov import POVNotReady, demote_spectated, detect_pov
from .redbox import RedboxNotReady, detect_redbox
from .renderer import make_srt, render
from .reviewer import ReviewNotReady, review_timeline
from .selftest import compare, generate
from .skull import SkullNotReady, detect_skull
from .timeline import build_timeline
from .util import ToolError, fmt_time, log, probe, run_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
DEFAULT_CFG = ROOT / "config.json"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _merge_recall_events(*channels: list[dict], dedup_sec: float = 1.5) -> list[dict]:
    """Union of recall channels (model medal / heuristic ink / OCR feed text).

    The channels watch different HUD evidence of the same kill, so either can
    miss what the others see. Events of any source within `dedup_sec` count as
    the same kill: keep ONE event, take the max strength, and merge every
    annotation field (OCR killer/victim/streak, model p_kill) so downstream
    consumers see the union of what each channel knows.
    """
    prio = {"ocr": 0, "killcard": 0, "redbox": 0, "skull": 1, "model": 2, "visual": 3}  # richest info first
    out = sorted((e for ch in channels for e in ch), key=lambda e: e["t"])
    merged: list[dict] = []
    for e in out:
        if merged and abs(e["t"] - merged[-1]["t"]) <= dedup_sec:
            m = merged[-1]
            m["strength"] = max(m["strength"], e["strength"])
            for k, v in e.items():
                if k in ("t", "source", "strength"):
                    continue
                if k not in m or m[k] is None:
                    m[k] = v
            m["channels"] = sorted(set(m.get("channels", [m["source"]])) | {e["source"]}, key=prio.get)
            continue
        keep = dict(e)
        merged.append(keep)
    for m in merged:
        m.setdefault("channels", [m["source"]])
    return merged


def process_video(video: str | Path, args, cfg: dict,
                  work_dir: str | Path | None = None,
                  out_path: str | Path | None = None) -> dict:
    video = Path(video).resolve()
    if not video.is_file():
        raise ToolError(f"视频文件不存在: {video}")
    meta = probe(video)
    log(f"{video.name}  {meta['width']}x{meta['height']}  {fmt_time(meta['duration'])}  "
        f"音轨={'有' if meta['has_audio'] else '无'}", "probe")

    work_dir = (Path(work_dir) if work_dir else OUTPUTS / video.stem).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path).resolve() if out_path else (work_dir / "highlights.mp4")

    rec = cfg.get("recognizer", {})
    visual = None
    if rec.get("backend") == "crispy":
        try:
            visual = detect_model(video, cfg, meta)
        except ModelNotReady as exc:
            log(f"神经网络识别器不可用：{exc}", "warn")
            log("本次回退启发式检测（识别器失败不影响出片）", "warn")
    model_events = list(visual["events"]) if visual is not None else []
    if visual is not None:
        # 双通道召回：启发式盯击杀播报(kill feed) ROI，与模型盯击杀奖牌框互补。
        # 模型漏检（如奖牌被遮挡/显示过短）时由启发式兜底，反之亦然。
        heur = detect_visual(video, cfg, meta)
        before = len(visual["events"])
        visual["events"] = _merge_recall_events(model_events, heur["events"])
        log(f"双通道召回合并：模型 {before} + 启发式 {len(heur['events'])} → "
            f"{len(visual['events'])} 个事件", "model")
    else:
        visual = detect_visual(video, cfg, meta)

    # OCR 第三通道：逐帧读击杀播报文字，拿到 谁杀了谁/连杀数 的确定性信息。
    # mode=full 时作为独立召回通道并入事件流；mode=annotate（默认）在时间轴
    # 构建后只对入选片段 OCR（连杀判定只发生在候选段内，成本大幅降低）。
    ocr = None
    ocr_cfg = cfg.get("ocr", {})
    if ocr_cfg.get("enabled") and ocr_cfg.get("mode", "annotate") == "full":
        try:
            ocr = detect_ocr(video, cfg, meta, model_events=model_events,
                             gate_events=visual["events"])
        except OCRNotReady as exc:
            log(f"OCR 通道不可用：{exc}", "warn")
            log("本次跳过 OCR 识别（不影响出片；安装依赖后重启生效）", "warn")
        if ocr is not None and ocr["events"]:
            before = len(visual["events"])
            visual["events"] = _merge_recall_events(visual["events"], ocr["events"])
            multi = [k for k in ocr["kills"] if k["streak"] >= 2]
            log(f"OCR 通道合并：+{len(ocr['events'])} 条具名击杀 → {len(visual['events'])} 个事件"
                + (f"（{len(multi)} 次连杀，最高 {max(k['streak'] for k in ocr['kills'])} 连杀）"
                   if ocr["kills"] else ""), "ocr")

    audio = detect_audio(video, cfg, has_audio=meta["has_audio"])
    if audio.get("skipped"):
        log(f"音频通道跳过: {audio['skipped']}", "warn")

    # OW 专属第四通道：屏幕中心击杀确认骷髅头（只有击杀者本人能看到）。
    # 眼窝双孔签名 + 生存期过滤；命中击杀播报出现新行时升级为确认玩家击杀。
    if cfg.get("skull", {}).get("enabled"):
        try:
            skull = detect_skull(video, cfg, meta)
            skull_events = skull["events"]
        except SkullNotReady as exc:
            log(f"骷髅头通道不可用：{exc}", "warn")
            log("本次跳过骷髅头识别（不影响出片；安装依赖后重启生效）", "warn")
            skull_events = []
        if skull_events:
            confirmed = [e for e in skull_events if e.get("confirmed")]
            before = len(visual["events"])
            visual["events"] = _merge_recall_events(visual["events"], skull_events)
            log(f"骷髅头通道合并：+{len(skull_events)} 个候选（{len(confirmed)} 个确认玩家击杀）"
                f"→ {len(visual['events'])} 个事件", "skull")

    # CS2 专属第五通道：底部中央回合击杀卡牌（顶卡数字=本回合玩家击杀数）。
    # 只在其他召回通道的事件时刻前后读数做差分：数字只随玩家自己的击杀增长。
    kc_events: list[dict] = []
    if cfg.get("killcard", {}).get("enabled") and cfg.get("recognizer", {}).get("game") == "csgo2":
        try:
            kc = detect_killcard(video, cfg, meta, gate_events=visual["events"])
            kc_events = kc["events"]
        except KillcardNotReady as exc:
            log(f"击杀卡牌通道不可用：{exc}", "warn")
            log("本次跳过击杀卡牌识别（不影响出片；安装依赖后重启生效）", "warn")

    # CS2 专属第六通道：右上击杀播报红框（当前视角玩家的击杀行整行红框）。
    # 与击杀卡牌同一批门控时刻读数，纯像素无 OCR；红框+卡牌变化=视角玩家的击杀。
    rb_events: list[dict] = []
    if cfg.get("redbox", {}).get("enabled") and cfg.get("recognizer", {}).get("game") == "csgo2":
        try:
            rb = detect_redbox(video, cfg, meta, gate_events=visual["events"])
            rb_events = rb["events"]
        except RedboxNotReady as exc:
            log(f"击杀红框通道不可用：{exc}", "warn")
            log("本次跳过击杀红框识别（不影响出片；安装依赖后重启生效）", "warn")

    # 红框×卡牌互证（合并前在各自事件上打标，合并只搬运结果）：
    # 卡牌有差分但无红框 → 撤销确认（可能是误读/非视角玩家击杀）；
    # 有红框但卡牌无差分 → 只作召回不确认（红框可持续 5-10s，避免滞留框虚增击杀）。
    # 红框通道全程零命中时视为通道失效，不做互证降级（保持旧行为）。
    if kc_events and rb_events:
        for e in kc_events:
            if e.get("confirmed") and not any(abs(e["t"] - r["t"]) <= 3.5 for r in rb_events):
                e["confirmed"] = False
                e["no_redbox"] = True
        for e in rb_events:
            if not any(abs(e["t"] - k["t"]) <= 2.5 for k in kc_events):
                e["confirmed"] = False
        demoted = sum(1 for e in kc_events if e.get("no_redbox"))
        if demoted:
            log(f"红框×卡牌互证：{demoted} 组卡牌事件无红框佐证，已撤销确认", "redbox")

    # CS2 专属第七通道：视角判定。死亡观战/GOTV 期间红框与卡牌跟随"被观看者"，
    # 据此撤销观战他人视角的击杀确认，并把整段观战区间从成片中剔除。
    # 注意必须放在互证之后、两通道合并之前：互证/视角都改源事件，合并只搬运。
    pov_info: dict = {"windows": [], "anchor_views": {}, "owner": "",
                      "other_pov": [], "reads": 0}
    if cfg.get("pov", {}).get("enabled") and cfg.get("recognizer", {}).get("game") == "csgo2":
        anchors = [e for e in list(kc_events) + list(rb_events) if e.get("confirmed")]
        try:
            pov_info = detect_pov(video, cfg, meta, anchor_events=anchors)
        except POVNotReady as exc:
            log(f"视角通道不可用：{exc}", "warn")
            log("本次跳过视角判定（不影响出片；安装依赖后重启生效）", "warn")

    if kc_events:
        before = len(visual["events"])
        visual["events"] = _merge_recall_events(visual["events"], kc_events)
        total = sum(e["kills"] for e in kc_events)
        log(f"击杀卡牌通道合并：+{len(kc_events)} 组玩家击杀（共 {total} 杀）"
            f"→ {len(visual['events'])} 个事件", "killcard")
    if rb_events:
        before = len(visual["events"])
        visual["events"] = _merge_recall_events(visual["events"], rb_events)
        log(f"击杀红框通道合并：+{len(rb_events)} 组红框击杀 → {len(visual['events'])} 个事件",
            "redbox")

    if pov_info.get("windows"):
        demoted = demote_spectated(visual["events"], pov_info)
        if demoted:
            names = sorted({e.get("spectate", "?") for e in visual["events"]
                            if e.get("spectate")})
            log(f"视角判定：撤销 {demoted} 组观战视角击杀确认"
                f"（被观战者：{'、'.join(names)}）", "pov")

    # HUD 私有确认事件（骷髅头/击杀卡牌）：时刻即"录像玩家本人击杀"的铁证，
    # 供 OCR annotate 玩家名推断与 VLM 玩家提示使用
    confirmed_events = [e for e in visual["events"] if e.get("confirmed")]

    _write_csv(work_dir / "signals.csv", visual["signals"])
    _write_csv(work_dir / "audio_rms.csv", audio["rms"])
    if ocr is not None:
        _write_csv(work_dir / "ocr_signal.csv", ocr["signals"])

    tl = build_timeline(video, visual["events"], audio["events"], meta["duration"], cfg,
                        ocr=ocr, exclude_intervals=pov_info.get("other_pov"))
    if pov_info.get("windows") or pov_info.get("owner"):
        tl["pov"] = {"owner": pov_info.get("owner", ""),
                     "windows": pov_info.get("windows", []),
                     "other_pov": [list(iv) for iv in pov_info.get("other_pov", [])],
                     "spectated_kills": {str(e["t"]): e.get("spectate", "")
                                         for e in visual["events"] if e.get("spectate")}}

    # annotate 模式：时间轴定型后，只对入选片段读击杀播报（默认模式）
    if ocr_cfg.get("enabled") and ocr_cfg.get("mode", "annotate") == "annotate":
        try:
            ocr = annotate_selected(video, cfg, meta, tl, model_events=model_events,
                                    confirmed_events=confirmed_events)
        except OCRNotReady as exc:
            log(f"OCR 通道不可用：{exc}", "warn")
            log("本次跳过 OCR 识别（不影响出片；安装依赖后重启生效）", "warn")
        else:
            _write_csv(work_dir / "ocr_signal.csv", ocr["signals"])
    if ocr is not None and ocr.get("player"):
        tl["player"] = ocr["player"]
    elif not tl.get("player"):
        # OCR 没跑/没读到名字：骷髅头确认事件自带玩家名（读播报行所得）
        named = [e.get("killer") for e in confirmed_events if e.get("killer")]
        if named:
            tl["player"] = max(set(named), key=named.count)
    if cfg.get("review", {}).get("enabled"):
        try:
            review_timeline(video, tl, cfg, work_dir)
        except ReviewNotReady as exc:
            log(f"语义复核不可用：{exc}", "warn")
            log("本次跳过复核，按纯信号模式出片（不影响成片）", "warn")
    (work_dir / "timeline.json").write_text(
        json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
    (work_dir / "captions.srt").write_text(make_srt(tl["selected"]), encoding="utf-8")
    log(f"候选 {tl['cluster_count']} 段，入选 {tl['selected_count']} 段 "
        f"（{tl['total_selected_sec']:.1f}s）", "timeline")

    render_info = None
    if getattr(args, "dry_run", False):
        log("dry-run：跳过视频渲染，只输出检测与决策产物", "render")
    else:
        render_info = render(video, tl["selected"], out_path, work_dir, cfg, meta)

    agent_info = orchestrate(tl, cfg, work_dir, render_info)
    (work_dir / "timeline.json").write_text(
        json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"work_dir": str(work_dir), "timeline": tl, "render": render_info,
            "agent": agent_info, "visual_events": visual["events"], "meta": meta}


def _summary(res: dict) -> None:
    tl = res["timeline"]
    print("\n" + "=" * 62)
    print("  完成  |  OW Highlight Agent P0")
    print("=" * 62)
    if res["render"]:
        print(f"  成片     : {res['render']['output']}")
    print(f"  决策报告 : {res['agent']['report']}")
    print(f"  时间轴   : {Path(res['work_dir']) / 'timeline.json'}")
    print(f"  信号数据 : {Path(res['work_dir']) / 'signals.csv'}")
    print(f"  字幕     : {Path(res['work_dir']) / 'captions.srt'}")
    print(f"  入选片段 : {tl['selected_count']} 段 / 共 {tl['total_selected_sec']:.1f}s")
    for seg in tl["selected"]:
        print(f"    #{seg.get('rank')}  {fmt_time(seg['start'])} → {fmt_time(seg['end'])}  "
              f"score={seg['score']:.3f}  {seg['reason']}")
    print("=" * 62 + "\n")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="配置文件路径（默认 ./config.json）")
    p.add_argument("--roi", help="ROI 比例 x,y,w,h，例如 0.72,0.05,0.27,0.22")
    p.add_argument("--threshold", type=float, help="事件判定阈值（默认 0.55，越低越敏感）")
    p.add_argument("--fps", type=float, help="采样帧率（默认 8）")
    p.add_argument("--top-k", type=int, help="最多入选片段数（默认 6）")
    p.add_argument("--pre", type=float, help="事件前留白秒数（默认 4）")
    p.add_argument("--post", type=float, help="事件后留白秒数（默认 3）")
    p.add_argument("--gap", type=float, help="相邻事件合并间隔秒数（默认 5）")
    p.add_argument("--no-audio", action="store_true", help="禁用音频通道")
    p.add_argument("--recognizer", choices=["none", "crispy"],
                   help="识别后端：none=启发式（默认）/ crispy=Crispy 预训练神经网络")
    p.add_argument("--game", choices=["overwatch", "csgo2", "valorant"],
                   help="游戏类型（默认 overwatch）：决定 crispy 模型权重、VLM 复核提示词、"
                        "启发式默认 ROI")
    p.add_argument("--confidence", type=float,
                   help="crispy 后端的击杀概率阈值（默认 0.6，漏检调低误检调高）")
    p.add_argument("--review", action="store_true",
                   help="启用 L2 语义复核（VLM 看关键帧判定事件类型/精彩度，需 API Key）")
    p.add_argument("--no-review", action="store_true",
                   help="强制关闭语义复核（覆盖 config.json 里的 review.enabled）")
    p.add_argument("--ocr", action="store_true",
                   help="启用 OCR 通道：逐帧读击杀播报文字，拿到击杀者/被杀者/连杀数"
                        "（离线 rapidocr，需 requirements-ocr.txt）")
    p.add_argument("--no-ocr", action="store_true",
                   help="强制关闭 OCR 通道（覆盖 config.json 里的 ocr.enabled）")
    p.add_argument("--player", help="你的游戏 ID（OCR 连杀判定与解说词用；默认自动推断）")
    p.add_argument("--vertical", action="store_true", help="输出 9:16 竖版 1080x1920")
    p.add_argument("--burn", action="store_true", help="烧录字幕（失败会自动跳过）")
    p.add_argument("--bgm", help="背景音乐文件路径")
    p.add_argument("--llm", action="store_true", help="启用 LLM 解说词（需设置 API Key）")
    p.add_argument("--dry-run", action="store_true", help="只检测与决策，不渲染视频")
    p.add_argument("--workdir", help="产物目录（默认 outputs/<视频名>）")
    p.add_argument("--out", help="成片输出路径（默认 <workdir>/highlights.mp4）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="守望先锋高光自动识别与剪辑 Agent · P0 本地原型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run.py demo                          一键生成合成录像并跑通全流程
  python run.py process D:/videos/match.mp4   处理自己的录像
  python run.py roi-preview D:/videos/match.mp4 --time 20
  python run.py process D:/videos/match.mp4 --threshold 0.4 --dry-run
  python run.py fetch-model                   下载 Crispy 预训练模型权重（一次性）
  python run.py process D:/videos/match.mp4 --recognizer crispy
  python run.py process D:/videos/match.mp4 --recognizer crispy --review   # +VLM 语义复核
""")
    parser.add_argument("--version", action="version", version=f"OW Highlight Agent {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="生成合成测试录像 → 检测 → 剪辑 → 报告 → 准确率对比")
    _add_common(p)

    p = sub.add_parser("self-test", help="只生成合成测试录像与真值文件")
    p.add_argument("--out-dir", default=str(SAMPLES), help="输出目录")
    p.add_argument("--config")

    p = sub.add_parser("probe", help="查看视频信息与建议 ROI")
    p.add_argument("video")
    _add_common(p)

    p = sub.add_parser("roi-preview", help="导出带 ROI 框的截图，用于标定")
    p.add_argument("video")
    p.add_argument("--time", type=float, default=3.0, help="截帧时间点（秒）")
    p.add_argument("--png", help="截图输出路径")
    _add_common(p)

    p = sub.add_parser("process", help="处理一段录像")
    p.add_argument("video")
    _add_common(p)

    p = sub.add_parser("fetch-model",
                      help="下载 Crispy 预训练模型权重（约 21 MB，一次性）")
    p.add_argument("--game", default="overwatch",
                   help="要下载权重的游戏，逗号分隔，或 all（默认 overwatch）")
    p.add_argument("--dest", help="权重目录（默认 models/crispy）")
    return parser


def dispatch(args, cfg: dict) -> int:
    if args.cmd == "self-test":
        gen = generate(args.out_dir)
        print(f"合成录像: {gen['video']}\n真值文件: {gen['truth']}")
        return 0

    if args.cmd == "fetch-model":
        games = [g.strip() for g in str(args.game).split(",") if g.strip()]
        dest = Path(args.dest).resolve() if args.dest else (ROOT / "models" / "crispy")
        fetch_weights(games, dest)
        print("\n完成。下一步:")
        print(f"  python run.py process <录像路径> --recognizer crispy --game {games[0]}")
        print("  （识别真实录像请先确认画面为 16:9 全屏，权重文件不会被提交到 git）")
        return 0

    if args.cmd == "probe":
        meta = probe(args.video)
        x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
        print(f"文件     : {Path(args.video).resolve()}")
        print(f"分辨率   : {meta['width']}x{meta['height']}  @ {meta['fps']:.2f}fps")
        print(f"时长     : {fmt_time(meta['duration'])}")
        print(f"音轨     : {'有' if meta['has_audio'] else '无'}")
        print(f"当前 ROI : 比例 {cfg['roi']['x']},{cfg['roi']['y']},{cfg['roi']['w']},{cfg['roi']['h']}")
        print(f"          像素 x={x} y={y} w={w} h={h}")
        rec = cfg.get("recognizer", {})
        game = rec.get("game", "overwatch")
        ready = (ROOT / "models" / "crispy" / f"{game}.npy").is_file()
        print(f"识别后端 : {rec.get('backend', 'none')}（{game}，权重{'已就绪' if ready else '未下载，可运行 fetch-model'}）")
        print(f"\n下一步: python run.py roi-preview \"{args.video}\" --time 20")
        return 0

    if args.cmd == "roi-preview":
        meta = probe(args.video)
        x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
        png = Path(args.png).resolve() if args.png else (OUTPUTS / f"{Path(args.video).stem}_roi.png")
        png.parent.mkdir(parents=True, exist_ok=True)
        vf = (f"drawbox=x={x}:y={y}:w={w}:h={h}:color=red@0.9:t=4,"
              f"drawbox=x={x}:y={y}:w={w}:h={h}:color=yellow@0.6:t=1")
        run_ffmpeg(["-ss", f"{float(args.time):.3f}", "-i", str(Path(args.video).resolve()),
                    "-frames:v", "1", "-vf", vf, str(png)])
        print(f"ROI 截图: {png}\n红框应完整覆盖右上角的击杀播报区域；不合适就用 --roi 调整。")
        return 0

    if args.cmd == "process":
        res = process_video(args.video, args, cfg,
                            work_dir=args.workdir, out_path=args.out)
        _summary(res)
        return 0

    if args.cmd == "demo":
        log("步骤 1/2：生成合成测试录像", "demo")
        gen = generate(SAMPLES)
        log("步骤 2/2：跑完整流水线", "demo")
        res = process_video(gen["video"], args, cfg,
                            work_dir=args.workdir, out_path=args.out)
        _summary(res)
        cmp = compare(res["visual_events"], gen["events"], tol=1.0)
        print("--- 合成真值对照（验证检测器是否工作） ---")
        print(f"  真值事件 {cmp['truth']} 个 / 检出 {cmp['detected']} 个 / 命中 {cmp['matched']} 个")
        print(f"  召回率 {cmp['recall'] * 100:.0f}%   精确率 {cmp['precision'] * 100:.0f}%")
        print("  （容差 ±1.0s；合成画面是理想信号，真实录像需要调 ROI 与阈值）\n")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = getattr(args, "config", None) or (str(DEFAULT_CFG) if DEFAULT_CFG.is_file() else None)
    try:
        cfg = load_config(cfg_path)
        # self-test needs no overrides; other commands take all common flags
        if args.cmd != "self-test":
            cfg = apply_overrides(cfg, args)
        return dispatch(args, cfg)
    except (ToolError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\n[错误] {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
