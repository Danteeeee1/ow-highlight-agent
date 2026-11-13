"""OW Highlight Agent - 桌面图形界面（纯 tkinter，无额外依赖）。

启动方式（任选其一）:
  1) 双击 start_gui.bat
  2) 命令行:  python gui.py

设计要点:
  * 流水线在后台 daemon 线程运行，界面不卡死；
  * 所有 print/log 输出经队列回传到日志窗口，实时滚动；
  * 处理完成后展示入选片段列表，可一键打开产物文件夹 / 播放成片；
  * 支持 启发式 / Crispy 神经网络 两种识别后端，含权重一键下载。
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- stdout 捕获
# pythonw 启动时 sys.stdout/stderr 为 None，任何 print 都会炸；
# 统一替换成写入队列的伪文件对象，GUI 主线程轮询取走。

_LOGQ: "queue.Queue[tuple[str, str]]" = queue.Queue()


class _QueueIO:
    """把 print/log 重定向进 GUI 日志队列。"""

    def __init__(self, tag: str, target):
        self.tag = tag
        self.target = target  # None（pythonw）或真流（调试用）

    def write(self, s: str) -> int:
        if s and s.strip():
            _LOGQ.put((self.tag, s))
        if self.target is not None:
            try:
                self.target.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        if self.target is not None:
            try:
                self.target.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


def _install_io_capture() -> None:
    sys.stdout = _QueueIO("out", sys.stdout if sys.stdout else None)
    sys.stderr = _QueueIO("err", sys.stderr if sys.stderr else None)


_install_io_capture()


def gui_log(msg: str, stage: str = "gui") -> None:
    """界面自身日志（也走队列，保证顺序一致）。"""
    _LOGQ.put(("gui", f"[{stage}] {msg}\n"))


# ---------------------------------------------------------------- 后台流水线
from owagent import __version__                     # noqa: E402
from owagent.cli import process_video, DEFAULT_CFG  # noqa: E402
from owagent.config import apply_overrides, load_config, pixel_roi  # noqa: E402
from owagent.selftest import compare, generate     # noqa: E402
from owagent.util import ToolError, fmt_time, probe  # noqa: E402
from owagent.model import fetch_weights, weights_dir  # noqa: E402
from owagent.reviewer import _resolve_key, ping_api  # noqa: E402


def _env_status() -> dict:
    """启动环境自检：ffmpeg / numpy / 模型权重。"""
    ok = {"ffmpeg": False, "ffprobe": False, "numpy": False, "weights": {}}
    import shutil
    ok["ffmpeg"] = shutil.which("ffmpeg") is not None
    ok["ffprobe"] = shutil.which("ffprobe") is not None
    try:
        import numpy  # noqa: F401
        ok["numpy"] = True
    except ImportError:
        pass
    for game in ("overwatch", "csgo2", "valorant"):
        ok["weights"][game] = (ROOT / "models" / "crispy" / f"{game}.npy").is_file()
    return ok


def _make_args(*, video: str = "", dry_run: bool = False, **kw) -> SimpleNamespace:
    """构造一个与 argparse 等价的参数对象供 process_video / apply_overrides 使用。"""
    base = dict(
        cmd="process", video=video, config=None, roi=None,
        threshold=None, fps=None, top_k=None, pre=None, post=None, gap=None,
        no_audio=False, recognizer=None, game=None, confidence=None,
        review=False, no_review=False,
        vertical=False, burn=False, bgm=None, llm=False,
        dry_run=dry_run, workdir=None, out=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- 界面
GAMES = {"overwatch": "守望先锋 Overwatch", "csgo2": "CS2", "valorant": "瓦罗兰特 Valorant"}
MAX_LOG_LINES = 8000


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"OW 高光剪辑 Agent v{__version__}")
        # 布局实测需求约 1160x800；默认给足空间，小屏（如 1366x768 笔记本）自动收缩
        w, h = 1200, 820
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        if sw < w + 40 or sh < h + 80:
            w, h = min(w, sw - 80), min(h, sh - 140)
        root.geometry(f"{w}x{h}")
        root.minsize(980, 680)
        try:
            root.iconbitmap(str(ROOT / "ow.ico")) if (ROOT / "ow.ico").is_file() else None
        except Exception:
            pass

        self.env = _env_status()
        self.running = False
        self.last_result: dict | None = None
        self._pending_after_check = None  # API 预检完成后的回调
        try:  # 启动时解析一次已有 Key（env / config.local.json），供输入框预填
            self._resolved_key = _resolve_key(
                load_config(DEFAULT_CFG if Path(DEFAULT_CFG).is_file() else None))
        except Exception:
            self._resolved_key = ""

        self._build_style()
        self._build_layout()
        self._refresh_env_labels()
        self._refresh_backend_state()
        self._refresh_review_state()
        self.root.after(100, self._drain_logq)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        gui_log("界面就绪。请选择一段录像，然后点「开始处理」。", "start")

    # ------------------------------------------------ style
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=10)
        style.configure("Big.TButton", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 10, "bold"))

    # ------------------------------------------------ layout
    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # ---- 1. 视频选择 ----
        vf = ttk.LabelFrame(outer, text=" 1 · 录像文件 ", padding=8)
        vf.pack(fill="x")
        self.var_video = tk.StringVar()
        ttk.Entry(vf, textvariable=self.var_video).pack(
            side="left", fill="x", expand=True, ipady=3)
        ttk.Button(vf, text="浏览…", command=self._pick_video).pack(side="left", padx=(6, 0))
        ttk.Button(vf, text="ROI 预览", command=self._roi_preview).pack(side="left", padx=(6, 0))
        ttk.Label(vf, text="截帧秒").pack(side="left", padx=(12, 2))
        self.var_snaptime = tk.StringVar(value="3")
        ttk.Spinbox(vf, from_=0, to=100000, width=6,
                    textvariable=self.var_snaptime).pack(side="left")
        self.lbl_meta = ttk.Label(vf, text="", foreground="#666")
        self.lbl_meta.pack(side="left", padx=(12, 0))

        # ---- 主区：左参数 / 右日志 ----
        paned = ttk.PanedWindow(outer, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=0)

        # ---- 2. 识别后端 ----
        bf = ttk.LabelFrame(left, text=" 2 · 识别后端 ", padding=8)
        bf.pack(fill="x")
        self.var_backend = tk.StringVar(value="none")
        ttk.Radiobutton(bf, text="启发式信号（免下载，开箱即用）", value="none",
                        variable=self.var_backend,
                        command=self._refresh_backend_state).pack(anchor="w")
        ttk.Radiobutton(bf, text="Crispy 神经网络（真实录像更准）", value="crispy",
                        variable=self.var_backend,
                        command=self._refresh_backend_state).pack(anchor="w")
        row = ttk.Frame(bf)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="游戏:").pack(side="left")
        self.var_game = tk.StringVar(value="overwatch")
        self.cb_game = ttk.Combobox(row, state="readonly", width=18, textvariable=self.var_game,
                                    values=list(GAMES.values()))
        self.cb_game.pack(side="left", padx=(4, 0))
        self.cb_game.bind("<<ComboboxSelected>>", lambda e: self._refresh_env_labels())
        row2 = ttk.Frame(bf)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="击杀置信度:").pack(side="left")
        self.var_conf = tk.DoubleVar(value=0.6)
        self.scale_conf = ttk.Scale(row2, from_=0.2, to=0.95,
                                    variable=self.var_conf, orient="horizontal")
        self.scale_conf.pack(side="left", fill="x", expand=True, padx=(6, 8))
        self.lbl_conf = ttk.Label(row2, width=4, text="0.60")
        self.lbl_conf.pack(side="left")
        self.var_conf.trace_add("write", lambda *_: self.lbl_conf.config(
            text=f"{self.var_conf.get():.2f}"))
        self.btn_fetch = ttk.Button(bf, text="下载模型权重（约 21 MB，一次性）",
                                    command=self._fetch_model)
        self.btn_fetch.pack(fill="x", pady=(6, 0))
        self.lbl_weights = ttk.Label(bf, text="", foreground="#0a7")
        self.lbl_weights.pack(anchor="w")

        # ---- 2b. 语义复核（VLM）----
        rf = ttk.LabelFrame(left, text=" 2b · 语义复核（VLM 大模型） ", padding=8)
        rf.pack(fill="x", pady=(8, 0))
        self.var_review = tk.BooleanVar(value=False)
        ttk.Checkbutton(rf, text="启用 DeepSeek 视觉复核（判断事件类型/精彩度并写解说）",
                        variable=self.var_review).pack(anchor="w")
        keyrow = ttk.Frame(rf)
        keyrow.pack(fill="x", pady=(6, 0))
        ttk.Label(keyrow, text="API Key:").pack(side="left")
        self.var_apikey = tk.StringVar(value=self._resolved_key)
        self.ent_apikey = ttk.Entry(keyrow, textvariable=self.var_apikey,
                                    show="•", width=26)
        self.ent_apikey.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.ent_apikey.bind("<Return>", lambda e: self._save_apikey())
        self.ent_apikey.bind("<FocusOut>", lambda e: self._save_apikey())
        self.var_showkey = tk.BooleanVar(value=False)
        ttk.Checkbutton(keyrow, text="显示", variable=self.var_showkey,
                        command=self._toggle_key_visibility).pack(side="left")
        ttk.Button(keyrow, text="测试连接", command=self._test_api).pack(
            side="left", padx=(4, 0))
        self.lbl_review = ttk.Label(rf, text="", justify="left", foreground="#0a7")
        self.lbl_review.pack(anchor="w", pady=(4, 0))
        ttk.Label(rf, text="Key 保存在本机 config.local.json（不会进 git）；"
                          "启用复核后，开始处理前会自动检测可用性",
                  foreground="#888", wraplength=360, justify="left").pack(anchor="w")

        # ---- 3. 检测与出片参数 ----
        pf = ttk.LabelFrame(left, text=" 3 · 参数 ", padding=8)
        pf.pack(fill="x", pady=(8, 0))
        grid = ttk.Frame(pf)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        def numrow(r, label, var, frm, to, inc, tip=None):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Spinbox(grid, from_=frm, to=to, increment=inc, width=8,
                        textvariable=var).grid(row=r, column=1, sticky="e")
            if tip:
                ttk.Label(grid, text=tip, foreground="#888").grid(
                    row=r, column=2, sticky="w", padx=(6, 0))

        self.var_threshold = tk.StringVar(value="0.55")
        numrow(0, "判定阈值", self.var_threshold, 0.05, 1.5, 0.05, "越低越敏感")
        self.var_topk = tk.StringVar(value="6")
        numrow(1, "最多片段数", self.var_topk, 1, 30, 1)
        self.var_pre = tk.StringVar(value="4")
        numrow(2, "事件前留白(秒)", self.var_pre, 0, 30, 0.5)
        self.var_post = tk.StringVar(value="3")
        numrow(3, "事件后留白(秒)", self.var_post, 0, 30, 0.5)
        self.var_maxseg = tk.StringVar(value="8")
        numrow(4, "VLM 复核段数", self.var_maxseg, 1, 20, 1, "每段一次调用")

        opts = ttk.Frame(pf)
        opts.pack(fill="x", pady=(6, 0))
        self.var_dry = tk.BooleanVar(value=False)
        self.var_vertical = tk.BooleanVar(value=False)
        self.var_burn = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="只检测不出片 (dry-run)",
                        variable=self.var_dry).pack(side="left")
        ttk.Checkbutton(opts, text="竖版 9:16", variable=self.var_vertical).pack(
            side="left", padx=(8, 0))
        ttk.Checkbutton(opts, text="烧录字幕", variable=self.var_burn).pack(
            side="left", padx=(8, 0))

        # ---- 环境状态 ----
        ef = ttk.LabelFrame(left, text=" 4 · 环境状态 ", padding=8)
        ef.pack(fill="x", pady=(8, 0))
        self.lbl_env = ttk.Label(ef, text="", justify="left")
        self.lbl_env.pack(anchor="w")

        left.columnconfigure(0, weight=1)

        # ---- 右侧：日志 + 结果 ----
        right = ttk.Frame(paned, padding=(0, 0, 0, 0))
        paned.add(right, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="运行日志", style="Section.TLabel").grid(
            row=0, column=0, sticky="w")
        self.txt_log = ScrolledText(right, height=16, state="disabled", wrap="word",
                                    font=("Consolas", 9), background="#1e1e1e",
                                    foreground="#d4d4d4", insertbackground="#d4d4d4")
        self.txt_log.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        for tag, color in (("err", "#f66"), ("warn", "#fb6"), ("gui", "#6cf"),
                           ("out", "#d4d4d4")):
            self.txt_log.tag_configure(tag, foreground=color)

        ttk.Label(right, text="入选高光片段", style="Section.TLabel").grid(
            row=2, column=0, sticky="w")
        cols = ("rank", "start", "end", "score", "event", "reason")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=7)
        for cid, text, w, anchor in (
                ("rank", "#", 36, "center"), ("start", "开始", 90, "center"),
                ("end", "结束", 90, "center"), ("score", "得分", 70, "center"),
                ("event", "事件/VLM", 110, "center"), ("reason", "判定依据", 340, "w")):
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=w, anchor=anchor, stretch=(cid == "reason"))
        self.tree.grid(row=3, column=0, sticky="nsew")
        right.rowconfigure(3, weight=0)

        # ---- 5. 操作条 ----
        bar = ttk.Frame(outer, padding=(0, 8, 0, 0))
        bar.pack(fill="x")
        self.btn_run = ttk.Button(bar, text="▶  开始处理", style="Big.TButton",
                                  command=self._run_process)
        self.btn_run.pack(side="left")
        self.btn_demo = ttk.Button(bar, text="🧪 生成演示并跑通全流程",
                                  command=self._run_demo)
        self.btn_demo.pack(side="left", padx=(8, 0))
        self.btn_open = ttk.Button(bar, text="📂 打开产物文件夹", state="disabled",
                                   command=self._open_folder)
        self.btn_open.pack(side="left", padx=(8, 0))
        self.btn_play = ttk.Button(bar, text="🎬 播放成片", state="disabled",
                                   command=self._play_output)
        self.btn_play.pack(side="left", padx=(8, 0))
        self.lbl_state = ttk.Label(bar, text="空闲", foreground="#0a7")
        self.lbl_state.pack(side="right")

    # ------------------------------------------------ 小工具
    def _cfg_with_overrides(self, **kw) -> dict:
        cfg = load_config(DEFAULT_CFG if Path(DEFAULT_CFG).is_file() else None)
        args = _make_args(
            recognizer=(self.var_backend.get() or None),
            game=([g for g, n in GAMES.items() if n == self.var_game.get()] or ["overwatch"])[0],
            confidence=self.var_conf.get(),
            threshold=float(self.var_threshold.get()),
            top_k=int(self.var_topk.get()),
            pre=float(self.var_pre.get()),
            post=float(self.var_post.get()),
            review=self.var_review.get(),
            vertical=self.var_vertical.get(),
            burn=self.var_burn.get(),
            **kw)
        cfg = apply_overrides(cfg, args)
        try:  # 复核段数非 CLI 参数，直接落到配置
            cfg["review"]["max_segments"] = max(1, int(self.var_maxseg.get()))
        except (ValueError, KeyError):
            pass
        return cfg, args

    def _game_key(self) -> str:
        return [g for g, n in GAMES.items() if n == self.var_game.get()] or ["overwatch"]

    def _set_running(self, on: bool, state_text: str | None = None):
        self.running = on
        for b in (self.btn_run, self.btn_demo, self.btn_fetch):
            b.config(state="disabled" if on else "normal")
        if state_text:
            self.lbl_state.config(text=state_text,
                                  foreground="#c60" if on else "#0a7")

    def _spawn(self, fn, done_msg: str):
        """启动后台任务。约定：工作线程绝不触碰 Tk 对象，只向队列发信号。"""
        def runner():
            try:
                fn()
                gui_log(done_msg, "done")
            except ToolError as exc:
                _LOGQ.put(("err", f"[错误] {exc}\n"))
            except Exception:
                _LOGQ.put(("err", "[异常] " + traceback.format_exc(limit=6) + "\n"))
            finally:
                _LOGQ.put(("sig", ("done", None)))
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    # ------------------------------------------------ 事件处理
    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="选择录像文件",
            filetypes=[("视频", "*.mp4 *.mkv *.mov *.avi *.flv *.ts"),
                       ("所有文件", "*.*")])
        if path:
            self.var_video.set(path)
            self._probe_video(path)

    def _probe_video(self, path: str):
        def fn():
            try:
                meta = probe(path)
                _LOGQ.put(("gui",
                           f"[probe] {meta['width']}x{meta['height']}  "
                           f"{fmt_time(meta['duration'])}  "
                           f"音轨={'有' if meta['has_audio'] else '无'}\n"))
                info = (f"{meta['width']}x{meta['height']} · "
                        f"{fmt_time(meta['duration'])} · "
                        f"{'有音轨' if meta['has_audio'] else '无音轨'}")
                _LOGQ.put(("sig", ("meta", info)))
            except Exception as exc:
                _LOGQ.put(("sig", ("meta", "")))
                _LOGQ.put(("err", f"[probe] 无法读取视频信息: {exc}\n"))
        threading.Thread(target=fn, daemon=True).start()

    def _roi_preview(self):
        video = self.var_video.get().strip()
        if not video:
            messagebox.showinfo("ROI 预览", "请先选择录像文件。")
            return
        if not self.env["ffmpeg"]:
            messagebox.showerror("ROI 预览", "未找到 ffmpeg，无法截图。请先安装 FFmpeg。")
            return
        try:
            t = float(self.var_snaptime.get())
        except ValueError:
            t = 3.0

        def fn():
            try:
                from owagent.util import run_ffmpeg
                cfg = load_config(DEFAULT_CFG if Path(DEFAULT_CFG).is_file() else None)
                meta = probe(video)
                x, y, w, h = pixel_roi(cfg, meta["width"], meta["height"])
                png = ROOT / "outputs" / f"{Path(video).stem}_roi.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                vf = (f"drawbox=x={x}:y={y}:w={w}:h={h}:color=red@0.9:t=4,"
                      f"drawbox=x={x}:y={y}:w={w}:h={h}:color=yellow@0.6:t=1")
                run_ffmpeg(["-ss", f"{t:.3f}", "-i", video,
                            "-frames:v", "1", "-vf", vf, str(png)])
                _LOGQ.put(("gui", f"[roi] 截图已生成: {png}\n"))
                _LOGQ.put(("sig", ("png", (str(png), (x, y, w, h)))))
            except Exception as exc:
                _LOGQ.put(("err", f"[roi] 截图失败: {exc}\n"))
        threading.Thread(target=fn, daemon=True).start()

    def _show_png(self, png: str, box=None):
        png = Path(png)
        win = tk.Toplevel(self.root)
        win.title(f"ROI 预览 - {png.name}")
        try:
            img = tk.PhotoImage(master=win, file=str(png))
            # 按宽度缩到 ~900px 内（PhotoImage.subsample 只支持整数倍）
            k = max(1, img.width() // 900 + 1)
            img = img.subsample(k, k)
            lbl = ttk.Label(win, image=img)
            lbl.image = img  # 防垃圾回收
            lbl.pack()
        except Exception as exc:
            ttk.Label(win, text=f"无法内嵌显示图片（{exc}），已改用系统看图工具打开。").pack()
            import os
            os.startfile(str(png))  # noqa: PTH
            return
        tip = "红框应完整覆盖右上角击杀播报区域；不合适时调整 config.json 的 roi 或用 --roi 参数。"
        if box:
            tip = f"检测框像素: x={box[0]} y={box[1]} w={box[2]} h={box[3]}  |  " + tip
        ttk.Label(win, text=tip, foreground="#666", padding=6).pack()

    def _fetch_model(self):
        game = self._game_key()
        dest = ROOT / "models" / "crispy"
        self._set_running(True, "下载模型权重…")
        gui_log(f"开始下载 Crispy 权重（{', '.join(game)}）…", "model")

        def fn():
            fetch_weights(game, dest)
            _LOGQ.put(("sig", ("env", None)))
        self._spawn(fn, "模型权重下载完成，可切换到「Crispy 神经网络」开始处理。")

    def _run_process(self):
        video = self.var_video.get().strip()
        if not video:
            messagebox.showinfo("开始处理", "请先选择录像文件（或点「生成演示」体验流程）。")
            return
        if not Path(video).is_file():
            messagebox.showerror("开始处理", f"文件不存在:\n{video}")
            return
        if not (self.env["ffmpeg"] and self.env["ffprobe"]):
            messagebox.showerror(
                "开始处理", "未找到 ffmpeg / ffprobe。\n请安装: winget install --id Gyan.FFmpeg -e\n"
                "安装后重新打开本程序。")
            return
        if self.var_backend.get() == "crispy" and not self.env["weights"].get(self._game_key()[0]):
            if not messagebox.askyesno(
                    "开始处理",
                    f"尚未下载 {self._game_key()[0]} 的模型权重，本次将回退启发式检测。\n"
                    "要先下载权重吗？（约 21 MB）"):
                pass
            else:
                self._fetch_model()
                return
        try:
            cfg, args = self._cfg_with_overrides(video=video,
                                                  dry_run=self.var_dry.get())
        except (ValueError, KeyError) as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._save_apikey()  # 先把输入框里的 Key 落盘/生效，再做检测
        if cfg["review"].get("enabled"):
            self._precheck_api_then(cfg, lambda: self._start_process(video, args, cfg))
            return
        self._start_process(video, args, cfg)

    def _start_process(self, video: str, args, cfg: dict):
        self._set_running(True, "处理中…（关闭窗口不会安全中止，请耐心等待）")
        self.tree.delete(*self.tree.get_children())
        self.btn_open.config(state="disabled")
        self.btn_play.config(state="disabled")
        gui_log(f"开始处理: {video}" + ("（dry-run）" if self.var_dry.get() else ""), "run")

        def fn():
            res = process_video(video, args, cfg)
            _LOGQ.put(("sig", ("result", res)))
        self._spawn(fn, "处理完成。")

    def _run_demo(self):
        if not (self.env["ffmpeg"] and self.env["ffprobe"]):
            messagebox.showerror("演示", "未找到 ffmpeg / ffprobe，请先安装 FFmpeg。")
            return
        try:
            cfg, _ = self._cfg_with_overrides(dry_run=self.var_dry.get())
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._save_apikey()
        if cfg["review"].get("enabled"):
            self._precheck_api_then(cfg, lambda: self._start_demo(cfg))
            return
        self._start_demo(cfg)

    def _start_demo(self, cfg: dict):
        self._set_running(True, "演示运行中…")
        self.tree.delete(*self.tree.get_children())
        gui_log("步骤 1/2：生成合成测试录像…", "demo")

        def fn():
            gen = generate(ROOT / "samples")
            gui_log(f"合成录像: {gen['video']}", "demo")
            gui_log("步骤 2/2：跑完整流水线…", "demo")
            args = _make_args(video=str(gen["video"]), dry_run=self.var_dry.get())
            res = process_video(gen["video"], args, cfg)
            _LOGQ.put(("sig", ("result", res)))
            cmp = compare(res["visual_events"], gen["events"], tol=1.0)
            _LOGQ.put(("gui",
                       f"[demo] 真值 {cmp['truth']} 个 / 检出 {cmp['detected']} 个 / "
                       f"命中 {cmp['matched']} 个 → 召回率 {cmp['recall']*100:.0f}%  "
                       f"精确率 {cmp['precision']*100:.0f}%\n"))
        self._spawn(fn, "演示完成。")

    def _on_done(self, res: dict):
        self.last_result = res
        tl = res["timeline"]
        for seg in tl.get("selected", []):
            rv = seg.get("review") or {}
            event = ""
            if rv.get("event_label"):
                event = (rv["event_label"]
                         + (f"×{rv['kill_count']}" if rv.get("kill_count") else "")
                         + f" {rv.get('excitement', '-')}/10")
            self.tree.insert("", "end", values=(
                seg.get("rank", ""), fmt_time(seg["start"]), fmt_time(seg["end"]),
                f"{seg['score']:.3f}", event, seg.get("reason", "")))
        if not tl.get("selected"):
            self.tree.insert("", "end", values=("", "-", "-", "-", "",
                            "没有片段入选：试着调低判定阈值，或检查 ROI 是否覆盖击杀播报区"))
        self.btn_open.config(state="normal")
        out = (res.get("render") or {}).get("output")
        self.btn_play.config(state="normal" if out else "disabled")
        _LOGQ.put(("gui",
                   f"[done] 入选 {tl.get('selected_count', 0)} 段，"
                   f"共 {tl.get('total_selected_sec', 0):.1f}s；"
                   f"产物目录 {res['work_dir']}\n"))

    def _open_folder(self):
        import os
        target = (self.last_result or {}).get("work_dir") or str(ROOT / "outputs")
        Path(target).mkdir(parents=True, exist_ok=True)
        os.startfile(target)  # noqa: PTH

    def _play_output(self):
        import os
        out = (self.last_result or {}).get("render", {}) or {}
        out = out.get("output")
        if out and Path(out).is_file():
            os.startfile(out)  # noqa: PTH

    # ------------------------------------------------ 状态刷新
    def _refresh_env_labels(self):
        # 权重状态实时读盘（下载完成后无需重启）
        for game in ("overwatch", "csgo2", "valorant"):
            self.env["weights"][game] = (ROOT / "models" / "crispy" / f"{game}.npy").is_file()
        lines = []
        lines.append(("✓" if self.env["ffmpeg"] else "✗") + " ffmpeg   "
                     + ("✓" if self.env["ffprobe"] else "✗") + " ffprobe   "
                     + ("✓" if self.env["numpy"] else "✗") + " numpy")
        g = self._game_key()[0]
        w = self.env["weights"].get(g, False)
        lines.append(("✓ 已就绪" if w else "✗ 未下载") + f" · 当前游戏权重: {g}.npy")
        missing = [k for k, v in self.env["weights"].items() if not v]
        if missing:
            lines.append(f"  （{', '.join(missing)} 权重未下载，可点下方按钮）")
        self.lbl_env.config(text="\n".join(lines),
                            foreground=("#333" if all(
                                (self.env["ffmpeg"], self.env["ffprobe"], self.env["numpy"]))
                                else "#a00"))

    # ------------------------------------------------ API Key 与预检
    def _toggle_key_visibility(self):
        self.ent_apikey.config(show="" if self.var_showkey.get() else "•")

    def _effective_key(self) -> str:
        """输入框内容优先，其次环境变量 / config.local.json 里已配置的 Key。"""
        return self.var_apikey.get().strip() or self._resolved_key

    def _save_apikey(self):
        """把输入框中的 Key 持久化到 config.local.json 并设会话环境变量（立即生效）。"""
        key = self.var_apikey.get().strip()
        if not key:
            return
        try:
            f = ROOT / "config.local.json"
            data: dict = {}
            if f.is_file():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    data = {}
            if not isinstance(data, dict):
                data = {}
            if data.get("review_api_key") != key:
                data["review_api_key"] = key
                f.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")
                gui_log("API Key 已保存到 config.local.json", "review")
        except OSError as exc:
            gui_log(f"API Key 保存失败: {exc}", "warn")
        # 会话内立即生效（_resolve_key 环境变量优先级最高）
        os.environ["OWAGENT_REVIEW_KEY"] = key
        self._resolved_key = key
        self._refresh_review_state()

    def _test_api(self):
        """手动「测试连接」：后台 ping，结果写回状态行与日志。"""
        self._save_apikey()
        try:
            cfg, _ = self._cfg_with_overrides(video="")
        except (ValueError, KeyError):
            cfg = {}
        key = self._effective_key()
        if not key:
            self.lbl_review.config(text="✗ 请先输入 API Key（sk-…）", foreground="#b60")
            return
        self.lbl_review.config(text="正在测试 API 连接…", foreground="#666")

        def fn():
            ok, msg = ping_api(cfg, key)
            _LOGQ.put(("sig", ("apitest", (ok, msg))))
        threading.Thread(target=fn, daemon=True).start()

    def _precheck_api_then(self, cfg: dict, start):
        """启用语义复核时：先异步检测 API，通过（或用户选择跳过复核）后再 start。"""
        key = self._effective_key()
        if not key:
            self._after_apicheck(False, "未配置 API Key", cfg, start)
            return
        self._set_running(True, "检测 API 连通性…")

        def fn():
            ok, msg = ping_api(cfg, key)
            _LOGQ.put(("sig", ("apicheck", (ok, msg, id(cfg)))))
        threading.Thread(target=fn, daemon=True).start()
        self._pending_after_check = (cfg, start)

    def _after_apicheck(self, ok: bool, msg: str, cfg: dict, start):
        gui_log(f"[review] 启动前 API 检测: {msg}", "review" if ok else "warn")
        if ok:
            start()
            return
        if messagebox.askyesno(
                "API 不可用",
                f"语义复核 API 检测失败：\n\n{msg}\n\n"
                "仍要继续处理吗？（将自动跳过语义复核，其余功能不受影响）"):
            cfg["review"]["enabled"] = False
            start()
        else:
            self._set_running(False, "空闲")

    def _refresh_review_state(self):
        """复核 Key 自检：输入框 > 环境变量 > config.local.json。"""
        key = self._effective_key()
        if key:
            self.lbl_review.config(
                text=f"✓ API Key 已配置（{key[:6]}…{key[-4:]}）· deepseek-flash",
                foreground="#0a7")
        else:
            self.lbl_review.config(
                text="✗ 未配置 API Key：在上方输入 DeepSeek Key（sk-…）",
                foreground="#b60")

    def _refresh_backend_state(self):
        crispy = self.var_backend.get() == "crispy"
        state = "readonly" if crispy else "disabled"
        self.cb_game.config(state=state)
        self.scale_conf.config(state="normal" if crispy else "disabled")
        w = self.env["weights"].get(self._game_key()[0], False)
        if crispy and not w:
            self.lbl_weights.config(
                text="⚠ 当前游戏权重未下载，处理时会回退启发式", foreground="#b60")
        else:
            self.lbl_weights.config(text="", foreground="#0a7")

    # ------------------------------------------------ 日志泵（唯一的 UI 更新入口）
    def _drain_logq(self):
        """主线程每 100ms 消费队列：文本进日志区，sig 分发到界面更新。"""
        try:
            for _ in range(200):
                tag, payload = _LOGQ.get_nowait()
                if tag == "sig":
                    kind, data = payload
                    if kind == "done":
                        self._set_running(False, "空闲")
                    elif kind == "result":
                        self._on_done(data)
                    elif kind == "meta":
                        self.lbl_meta.config(text=data)
                    elif kind == "png":
                        self._show_png(*data)
                    elif kind == "env":
                        self._refresh_env_labels()
                        self._refresh_backend_state()
                    elif kind == "apitest":
                        ok, msg = data
                        self.lbl_review.config(
                            text=("✓ " if ok else "✗ ") + msg,
                            foreground=("#0a7" if ok else "#b60"))
                        gui_log(f"API 连接测试: {msg}", "review" if ok else "warn")
                    elif kind == "apicheck":
                        ok, msg, cfg_id = data
                        pending = self._pending_after_check
                        self._pending_after_check = None
                        if pending is None:
                            continue
                        cfg, start = pending
                        if id(cfg) != cfg_id:  # 过期的检测结果（用户已发起新任务）
                            continue
                        self._after_apicheck(ok, msg, cfg, start)
                    continue
                text = payload
                self.txt_log.config(state="normal")
                cls = "err" if ("[错误]" in text or "[异常]" in text) else tag
                if "[warn]" in text:
                    cls = "warn"
                self.txt_log.insert("end", text, cls)
                # 截断过长日志
                if int(self.txt_log.index("end-1c").split(".")[0]) > MAX_LOG_LINES:
                    self.txt_log.delete("1.0", f"{MAX_LOG_LINES//4}.0")
                self.txt_log.config(state="disabled")
                self.txt_log.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_logq)

    def _on_close(self):
        if self.running:
            if not messagebox.askyesno(
                    "退出", "处理仍在后台运行，现在退出会中断它。\n确定退出吗？"):
                return
        self.root.destroy()


def main() -> int:
    # pythonw 无控制台：任何导入期/启动期异常都落到 error.log，方便排查黑启动
    try:
        root = tk.Tk()
    except Exception:
        (ROOT / "error.log").write_text(
            "无法创建 Tk 窗口:\n" + traceback.format_exc(), encoding="utf-8")
        raise
    try:
        App(root)
        root.mainloop()
    except Exception:
        (ROOT / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
