"""GUI 冒烟测试：无头实例化 App，自动跑 demo 全流程，校验结果链路。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
import gui


def main() -> int:
    root = tk.Tk()
    root.withdraw()  # 不打扰用户
    app = gui.App(root)

    # 1. 参数组装链路
    cfg, args = app._cfg_with_overrides(video="x.mp4", dry_run=False)
    assert cfg["recognizer"]["backend"] == app.var_backend.get() == "none"
    assert cfg["detect"]["threshold"] == float(app.var_threshold.get()) == 0.55
    assert cfg["clip"]["pre_roll"] == 4.0 and cfg["clip"]["post_roll"] == 3.0
    assert args.video == "x.mp4"
    print("[1] cfg_with_overrides ok")

    # 2. 切换到 crispy 后端的参数链路（独立 cfg，不覆盖步骤 1 的）
    app.var_backend.set("crispy")
    app._refresh_backend_state()
    cfg_crispy, _ = app._cfg_with_overrides(video="x.mp4")
    assert cfg_crispy["recognizer"]["backend"] == "crispy"
    assert cfg_crispy["recognizer"]["confidence"] == app.var_conf.get()
    app.var_backend.set("none")
    print("[2] crispy backend override ok")

    # 2b. 语义复核开关链路
    app.var_review.set(True)
    cfg_r, _ = app._cfg_with_overrides(video="x.mp4")
    assert cfg_r["review"]["enabled"] is True
    assert cfg_r["review"]["max_segments"] == 8
    app.var_review.set(False)
    cfg_r2, _ = app._cfg_with_overrides(video="x.mp4")
    assert cfg_r2["review"]["enabled"] is False
    print("[2b] review toggle ok")

    # 3. 跑 demo 全流程（后台线程，dry-run 模式与真实 GUI 路径一致走队列信号）
    app._set_running(True, "测试中")
    a = gui._make_args(video=str(gui.ROOT / "samples" / "nonexist.mp4"), dry_run=True)

    def fn():
        gen = gui.generate(gui.ROOT / "samples")
        res = gui.process_video(gen["video"], a, cfg)
        gui._LOGQ.put(("sig", ("result", res)))

    import threading
    threading.Thread(target=fn, daemon=True).start()

    deadline = time.time() + 240
    while app.last_result is None and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    assert app.last_result is not None, "demo 流程超时（未收到 result 信号）"
    print("[3] demo process finished via queue signal")

    # 4. 结果链路
    res = app.last_result
    assert res["timeline"]["selected_count"] >= 1, "应当至少入选 1 段"
    kids = app.tree.get_children()
    assert len(kids) == res["timeline"]["selected_count"], "Treeview 行数应与入选数一致"
    assert str(app.btn_open["state"]) == "normal"
    print(f"[4] result ok: {res['timeline']['selected_count']} segments, "
          f"work_dir={res['work_dir']}")

    # 5. 日志泵：手动注入 done 信号验证界面恢复空闲
    gui._LOGQ.put(("sig", ("done", None)))
    deadline = time.time() + 10
    while app.running and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    content = app.txt_log.get("1.0", "end")
    assert not app.running, "收到 done 信号后应恢复空闲"
    assert "timeline" in content, "日志区应有运行日志"
    print("[5] log panel + sig:done ok, sample:", content[:150].replace(chr(10), " | "))

    root.destroy()
    print("\nALL GUI SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
