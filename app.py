"""Streamlit local client for the OW Highlight Agent P0 prototype.

Run with:  python -m streamlit run app.py
The pipeline itself lives in owagent/ - this file only renders UI and forwards
widget values through owagent.webui.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from owagent import webui
from owagent.selftest import compare
from owagent.util import fmt_time

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="OW Highlight Agent", layout="wide")
st.title("守望先锋高光剪辑 · 本地工作台")
st.caption("P0 原型 | 完全离线 | 仅分析录屏像素与音轨，不读取游戏内存")

base = webui.load_base_config()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("检测参数")
    st.caption("改完直接点「开始处理」，无需保存")

    st.subheader("ROI 检测区域（画面比例）")
    a, b = st.columns(2)
    rx = a.number_input("x 左", 0.0, 1.0, float(base["roi"]["x"]), 0.01, format="%.2f")
    ry = b.number_input("y 上", 0.0, 1.0, float(base["roi"]["y"]), 0.01, format="%.2f")
    a, b = st.columns(2)
    rw = a.number_input("w 宽", 0.01, 1.0, float(base["roi"]["w"]), 0.01, format="%.2f")
    rh = b.number_input("h 高", 0.01, 1.0, float(base["roi"]["h"]), 0.01, format="%.2f")
    roi_time = st.number_input("ROI 预览截帧时间 (s)", 0.0, 100000.0, 3.0, 1.0)
    st.caption("默认框住 1080p 右上角击杀播报，务必先用「预览 ROI 框」确认")

    threshold = st.slider("事件判定阈值", 0.20, 0.90,
                          float(base["detect"]["threshold"]), 0.05,
                          help="漏检多就调低，误检多就调高")
    fps = st.slider("采样帧率", 2, 16, int(base["sample"]["fps"]), 1)

    with st.expander("片段切分"):
        pre = st.number_input("事件前留白 (s)", 0.0, 15.0, float(base["clip"]["pre_roll"]), 0.5)
        post = st.number_input("事件后留白 (s)", 0.0, 15.0, float(base["clip"]["post_roll"]), 0.5)
        gap = st.number_input("事件合并间隔 (s)", 0.5, 30.0, float(base["clip"]["merge_gap"]), 0.5)
        top_k = st.number_input("最多入选片段", 1, 20, int(base["select"]["top_k"]), 1)
        st.caption("合并间隔建议不小于 前留白 + 后留白，否则相邻片段会重叠")

    with st.expander("渲染与输出"):
        no_audio = st.checkbox("禁用音频通道", value=False)
        vertical = st.checkbox("输出 9:16 竖版", value=False)
        burn = st.checkbox("烧录字幕", value=False)
        use_llm = st.checkbox("启用 LLM 解说词（需 API Key）", value=False)
        dry_run = st.checkbox("只检测不渲染（快速调参）", value=False)
        bgm = st.text_input("BGM 文件路径（可选）", "")
        workdir = st.text_input("产物目录（留空自动）", "")

    st.divider()
    st.caption("配置文件 config.json 的取值作为控件默认值；这里只影响本次运行")

# ---------------------------------------------------------------- main
try:
    cfg = webui.build_config(
        base, roi=(rx, ry, rw, rh), threshold=threshold, fps=fps,
        pre=pre, post=post, gap=gap, top_k=top_k, no_audio=no_audio,
        vertical=vertical, burn=burn, bgm=bgm, llm=use_llm)
except ValueError as exc:
    st.error(f"参数不合法：{exc}")
    st.stop()

st.subheader("1. 选择视频")
source = st.radio("视频来源", ["本地路径", "上传文件", "合成 Demo（一键体验）"],
                  horizontal=True, label_visibility="collapsed")
path_text, uploaded = "", None
if source == "本地路径":
    path_text = st.text_input("视频绝对路径", placeholder=r"D:\videos\match.mp4",
                              help="支持 mp4 / mkv / mov / ts 等 FFmpeg 可解格式")
elif source == "上传文件":
    uploaded = st.file_uploader("选择录像文件", type=["mp4", "mkv", "mov", "ts", "flv", "avi"])
    st.caption("大录像建议改用「本地路径」，避免浏览器上传数 GB 文件")
else:
    st.info("点「开始处理」会自动生成 30s 合成测试录像（8 次模拟击杀播报），并给出召回率 / 精确率对照")

c1, c2, c3 = st.columns([1, 1, 2])
btn_probe = c1.button("探测视频")
btn_roi = c2.button("预览 ROI 框")
btn_run = c3.button("开始处理", type="primary")


def get_video() -> Path:
    if source.startswith("合成"):
        if not st.session_state.get("demo_video"):
            with st.spinner("正在生成合成测试录像…"):
                st.session_state["demo_video"] = webui.generate_demo()["video"]
        return Path(st.session_state["demo_video"])
    if source == "本地路径":
        p = Path(path_text.strip().strip('"').strip("'"))
        if not p.is_file():
            raise ValueError(f"文件不存在：{p}")
        return p
    if uploaded is None:
        raise ValueError("请先选择一个视频文件")
    dest = ROOT / "uploads" / uploaded.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = uploaded.getvalue()
    if not dest.exists() or dest.stat().st_size != len(data):
        dest.write_bytes(data)
    return dest


if btn_probe:
    try:
        v = get_video()
        meta = webui.probe_info(v)
        st.success("视频信息")
        st.json({"文件": str(v), "分辨率": f"{meta['width']}x{meta['height']}",
                 "帧率": round(meta["fps"], 2), "时长": fmt_time(meta["duration"]),
                 "音轨": "有" if meta["has_audio"] else "无"})
    except Exception as exc:
        st.error(f"探测失败：{exc}")

if btn_roi:
    try:
        v = get_video()
        png, box = webui.preview_roi(v, cfg, time_sec=roi_time)
        st.success(f"ROI 像素框：x={box[0]} y={box[1]} w={box[2]} h={box[3]}")
        st.image(png, caption="红框应完整覆盖击杀播报区域；不合适请调整左侧 ROI 参数")
    except Exception as exc:
        st.error(f"预览失败：{exc}")

if btn_run:
    st.session_state.pop("result", None)
    st.session_state["logs"] = []
    try:
        v = get_video()
        with st.status("正在处理录像…", expanded=True) as status:
            def on_log(line: str) -> None:
                st.session_state["logs"].append(line)
                status.write(line)

            res = webui.run_pipeline(v, cfg, dry_run=dry_run,
                                     work_dir=workdir.strip() or None,
                                     on_log=on_log)
            status.update(label="处理完成", state="complete", expanded=False)
        st.session_state["result"] = res
        st.session_state["cfg_used"] = cfg
        st.session_state["source_used"] = source
        st.success("处理完成，结果如下")
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        with st.expander("错误详情"):
            st.code(traceback.format_exc(), language="text")


def download_button(label: str, path: str, mime: str) -> None:
    p = Path(path) if path else Path("")
    if p.is_file():
        st.download_button(label, data=p.read_bytes(), file_name=p.name,
                           mime=mime)
    else:
        st.button(label, disabled=True)


res = st.session_state.get("result")
if res:
    cfg_used = st.session_state.get("cfg_used", cfg)
    source_used = st.session_state.get("source_used", source)
    tl = res["timeline"]

    st.divider()
    st.subheader("2. 处理结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("入选片段", tl["selected_count"])
    m2.metric("成片时长", f"{tl['total_selected_sec']:.1f} s")
    m3.metric("视觉事件", tl["event_count"]["visual"])
    m4.metric("音频事件", tl["event_count"]["audio"])

    truth_file = ROOT / "samples" / "demo_truth.json"
    if source_used.startswith("合成") and truth_file.is_file():
        truth = json.loads(truth_file.read_text(encoding="utf-8"))["events"]
        cmp_result = compare(res["visual_events"], truth)
        st.info(f"合成真值对照：命中 {cmp_result['matched']}/{cmp_result['truth']} · "
                f"召回率 {cmp_result['recall'] * 100:.0f}% · "
                f"精确率 {cmp_result['precision'] * 100:.0f}%（容差 ±1s）")

    if res["render"]:
        st.video(res["render"]["output"])
        st.caption(f"{res['render']['output']}  |  {res['render']['resolution']}")
    else:
        st.warning("dry-run 模式：未渲染视频，以下为检测与决策产物")

    rows = [{"#": s.get("rank"), "源片开始": fmt_time(s["start"]),
             "源片结束": fmt_time(s["end"]), "时长 (s)": s["duration"],
             "得分": s["score"], "判定理由": s["reason"]} for s in tl["selected"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("没有入选片段，请降低阈值或检查 ROI 后重试")

    sig_file = Path(res["work_dir"]) / "signals.csv"
    if sig_file.is_file():
        with st.expander("逐帧信号曲线（调阈值时看这里）"):
            frame = pd.read_csv(sig_file)
            if "t" in frame.columns:
                frame = frame.set_index("t")
                cols = [c for c in ("score", "ink_rise") if c in frame.columns]
                frame["threshold"] = cfg_used["detect"]["threshold"]
                st.line_chart(frame[cols + ["threshold"]])
                st.caption("score 越过 threshold 线且 ink_rise 出现尖峰时记为一次播报事件；"
                           "平静段 ink_rise 应接近 0，否则 ROI 可能框进了高亮背景。")

    st.subheader("3. 产物下载")
    wd = Path(res["work_dir"])
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        download_button("成片 MP4", res["render"]["output"] if res["render"] else "", "video/mp4")
    with d2:
        download_button("字幕 SRT", str(wd / "captions.srt"), "text/plain")
    with d3:
        download_button("时间轴 JSON", str(wd / "timeline.json"), "application/json")
    with d4:
        download_button("决策报告 MD", res["agent"]["report"], "text/markdown")

    report = Path(res["agent"]["report"])
    if report.is_file():
        with st.expander("查看决策报告"):
            st.markdown(report.read_text(encoding="utf-8"))

    logs = st.session_state.get("logs", [])
    if logs:
        with st.expander("运行日志"):
            st.code("\n".join(logs), language="text")

    if st.button("清空结果"):
        st.session_state.pop("result", None)
        st.session_state.pop("logs", None)
        st.rerun()

st.divider()
st.caption("命令行同样可用：python run.py process <video>  |  完整文档见 README.md 第 15 节")
