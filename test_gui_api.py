"""GUI API 输入栏 + 启动前检测 冒烟测试。

覆盖:
  1. ping_api 真实 Key（联网）→ ok
  2. ping_api 无效 Key → 401 识别
  3. _save_apikey 持久化回写 config.local.json + 环境变量（含还原）
  4. 启动前检测成功路径: _precheck_api_then → start 被调用
  5. 启动前检测失败路径: 用户选择"继续" → 复核被禁用 + start 被调用
  6. Key 显示/隐藏切换、状态行文案
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
import gui


def main() -> int:
    local = gui.ROOT / "config.local.json"
    backup = local.read_text(encoding="utf-8") if local.is_file() else None
    try:
        return _run(local)
    finally:  # 无论断言成败都还原 Key 文件
        if backup is not None:
            local.write_text(backup, encoding="utf-8")


def _run(local: Path) -> int:
    root = tk.Tk()
    root.withdraw()
    app = gui.App(root)

    # ---- 1. ping_api 真实 Key（联网） ----
    cfg, _ = app._cfg_with_overrides(video="x.mp4")
    real_key = app._effective_key()
    assert real_key.startswith("sk-"), "应能从 config.local.json 解析出 Key"
    ok, msg = gui.ping_api(cfg, real_key)
    assert ok, f"真实 Key 检测应通过: {msg}"
    print(f"[1] ping_api 真实 Key ok: {msg}")

    # ---- 2. ping_api 无效 Key ----
    ok2, msg2 = gui.ping_api(cfg, "sk-invalid-key-for-test")
    assert not ok2 and "401" in msg2, f"无效 Key 应报 401: {ok2}, {msg2}"
    print(f"[2] ping_api 无效 Key ok: {msg2}")

    # ---- 3. _save_apikey 持久化 + 还原 ----
    real_env = os.environ.get("OWAGENT_REVIEW_KEY", "")
    try:
        app.var_apikey.set("sk-dummy-000000000000")
        app._save_apikey()
        data = json.loads(local.read_text(encoding="utf-8"))
        assert data["review_api_key"] == "sk-dummy-000000000000"
        assert os.environ["OWAGENT_REVIEW_KEY"] == "sk-dummy-000000000000"
        # 还原真实 Key
        app.var_apikey.set(real_key)
        app._save_apikey()
        data = json.loads(local.read_text(encoding="utf-8"))
        assert data["review_api_key"] == real_key
        print("[3] _save_apikey 持久化与还原 ok")
    finally:
        if real_env:
            os.environ["OWAGENT_REVIEW_KEY"] = real_env
        else:
            os.environ.pop("OWAGENT_REVIEW_KEY", None)

    # ---- 4. 启动前检测 · 成功路径 ----
    app.var_review.set(True)
    cfg_ok, _ = app._cfg_with_overrides(video="x.mp4")
    assert cfg_ok["review"]["enabled"] is True
    called = []
    app._precheck_api_then(cfg_ok, lambda: called.append(("ok", cfg_ok["review"]["enabled"])))
    deadline = time.time() + 30
    while not called and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    assert called, "API 检测成功后应执行 start"
    assert called[0] == ("ok", True), "成功路径不应禁用复核"
    print("[4] 预检成功路径 ok（start 被调用，复核保持启用）")

    # ---- 5. 启动前检测 · 失败路径（用户选择继续） ----
    cfg_bad, _ = app._cfg_with_overrides(video="x.mp4")
    called2 = []
    gui.messagebox.askyesno = lambda *a, **k: True  # 模拟用户点"是"（跳过复核继续）
    app._after_apicheck(False, "API Key 无效或已过期（401）", cfg_bad,
                        lambda: called2.append(cfg_bad["review"]["enabled"]))
    assert called2 == [False], "失败+继续路径应禁用复核后再 start"
    print("[5] 预检失败路径 ok（复核被禁用，start 被调用）")

    # ---- 6. 显示/隐藏 + 状态行 ----
    assert app.ent_apikey.cget("show") == "•"
    app.var_showkey.set(True)
    app._toggle_key_visibility()
    assert app.ent_apikey.cget("show") == ""
    app.var_showkey.set(False)
    app._toggle_key_visibility()
    app.var_apikey.set("")
    app._resolved_key = ""
    app._refresh_review_state()
    assert "未配置" in app.lbl_review.cget("text")
    print("[6] Key 显示切换与状态行 ok")

    # 清空 pending，注入 done 恢复空闲，避免析构告警
    app._pending_after_check = None
    gui._LOGQ.put(("sig", ("done", None)))
    root.update()
    root.destroy()
    print("\nALL API GUI TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
