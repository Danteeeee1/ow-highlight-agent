# OW Highlight Agent · P0 本地原型

自动识别《守望先锋》录像里的**击杀播报/高光信号**，把候选片段打分、排序，再用 FFmpeg 剪成一段集锦，并输出时间轴、字幕和决策报告。

> **两条识别路线**：默认走「HUD 像素信号 + 音频能量」的零依赖启发式；可选启用 **Crispy 预训练神经网络**（第 16 节）——真实游戏画面训练的击杀检测模型，纯 numpy 推理，支持守望先锋 / CS2 / 瓦罗兰特。启发式路线**不认识英雄、不读文字、不理解击杀语义**，目的是用最小依赖把「采集 → 召回 → 打分 → 时间轴 → 剪辑 → 反馈」这条链路完整跑通。

**本机实测结果**（Windows 11 / Python 3.13.14 / FFmpeg 9.0.2）：

| 项目 | 结果 |
|---|---|
| 合成测试录像 | 30s，8 次模拟击杀播报 |
| 视觉事件召回 | 8 / 8，召回率 **100%**，精确率 **100%**（±1.0s 容差） |
| 音频峰值召回 | 8 / 8 |
| 成片 | 3 段无重叠片段、25.26s、H.264 跟随源分辨率/帧率 + AAC |
| 模型接入验证 | 预处理与 Crispy 原版**逐像素一致（diff=0）**；OW2 风格红色击杀标记触发 p_kill 0.01→0.99；真实 CS2 录像误报 0 |
| 额外验证 | 竖版 1080×1920、字幕烧录、BGM 混合、无音轨补静音、LLM 无 Key 降级、权重缺失自动回退启发式、dry-run 均通过 |

---

## 1. 能力边界

**能做**
- 从本地 mp4/录屏中召回「HUD 播报出现」与「音频突变」两类事件
- 可选：用 Crispy 预训练神经网络做**语义级击杀召回**（真实画面训练，第 16 节）
- 事件聚类成片段、多维打分、Top-K 选取、按源片时序重排成集锦
- 输出 MP4、SRT 字幕、时间轴 JSON、逐帧信号 CSV、中文决策报告
- 可选：9:16 竖版、烧录字幕、混入 BGM、调用 OpenAI 兼容 API 生成解说词

**不能做**（留给 P1/P2）
- OCR 读英雄名/击杀数、识别「双杀/团灭/全场最佳」文字
- 视觉语义判断（残血反杀、大招质量、走位精彩度）
- 读取游戏内存、解析回放文件、读取官方 telemetry
- 实时直播流处理（当前是离线批处理）

---

## 2. 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.8+（3.13.14 实测） | **不需要 pip install 任何包**，全部用标准库 |
| FFmpeg / FFprobe | 5.0+（9.0.2 实测） | 需带 `libx264`；烧录字幕还需 `libass`（官方 full build 都有） |
| 操作系统 | Windows / macOS / Linux | 路径处理已做跨平台适配 |

安装 FFmpeg：

```bash
# Windows（装完重开终端）
winget install --id Gyan.FFmpeg -e
# 或 choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

验证：

```bash
python --version
ffmpeg -version | head -1
ffprobe -version | head -1
```

> **命令行原型零 pip 依赖**。两条可选路线才需要装包：
> - 图形界面（第 15 节）：`pip install -r requirements.txt`
> - 神经网络识别后端（第 16 节）：`pip install -r requirements-model.txt`（numpy；CS2/瓦罗兰特另需 Pillow）

> Windows 若 `python` 命令不可用，把下文所有 `python` 换成 `py`。

---

## 3. 30 秒快速开始

不需要任何守望先锋录像，项目自带合成测试视频生成器：

```bash
cd ow-highlight-agent
python run.py demo
```

这一条命令会依次完成：生成 30s 合成录像 → 探测视频 → ROI 抽帧检测 → 音频峰值检测 → 片段合并打分 → FFmpeg 剪辑 → 生成报告 → 与真值对比。

预期输出（节选）：

```
[demo] 已生成合成测试录像 ...\samples\demo_match.mp4（30s，8 次模拟击杀播报）
[detect] ROI 像素框 x=230 y=8 w=86 h=38 → 采样 96x32 @ 8.0fps
[detect] 扫描 240 帧 / 30.0s，召回 8 个视觉事件
[detect] 音频 30.0s，召回 8 个峰值事件
[timeline] 候选 3 段，入选 3 段 （25.2s）
[render] 切片 1/3  00:00:00.00 → 00:00:08.00
[render] 成片已输出: ...\outputs\demo_match\highlights.mp4
--- 合成真值对照 ---
  召回率 100%   精确率 100%
```

打开 `outputs/demo_match/highlights.mp4` 就能看到成片，`outputs/demo_match/report.md` 是决策报告。

> 不想敲命令？用第 15 节的 Streamlit 图形界面：调参、ROI 预览、实时日志、成片播放和下载都在浏览器里完成。

---

## 4. 处理自己的录像：标准四步

```bash
# ① 看视频信息与当前 ROI 落点
python run.py probe "D:/videos/my_match.mp4"

# ② 导出一张带 ROI 框的截图，肉眼确认框住了右上角击杀播报
python run.py roi-preview "D:/videos/my_match.mp4" --time 20

# ③ 先 dry-run 只做检测和决策，不渲染（快），看召回数量是否合理
python run.py process "D:/videos/my_match.mp4" --threshold 0.45 --dry-run

# ④ 满意后正式出片
python run.py process "D:/videos/my_match.mp4" --threshold 0.45
```

**第 ② 步最关键**。默认 ROI 是按 OW2 1080p 击杀播报区域给的，但如果你的分辨率、UI 缩放、客户端语言或直播遮挡不同，一定要先标定。生成的 `outputs/<视频名>_roi.png` 里红框应完整覆盖播报条区域。

调整 ROI 用比例坐标 `--roi x,y,w,h`（0~1，相对整幅画面）：

```bash
python run.py process "D:/videos/my_match.mp4" --roi 0.70,0.04,0.29,0.24
```

调好后把值写进 `config.json` 的 `roi` 段，以后不用每次带参数。

---

## 5. ROI 标定详解

`roi` 四个值都是**画面比例**，所以同一份配置能适配任意分辨率：

| 字段 | 含义 | 默认 |
|---|---|---|
| `x` | 框左上角的横向位置 | `0.72` |
| `y` | 框左上角的纵向位置 | `0.05` |
| `w` | 框宽度 | `0.27` |
| `h` | 框高度 | `0.22` |

默认框在各分辨率下的实际像素（代码会自动偶数对齐以兼容 yuv420p）：

| 录像分辨率 | 像素框 x, y, w, h |
|---|---|
| 1920×1080 | 1382, 54, 518, 236 |
| 2560×1440 | 1842, 72, 690, 316 |
| 1280×720 | 920, 36, 344, 158 |
| 320×180（合成测试） | 230, 8, 86, 38 |

**怎么判断 ROI 对不对**

1. 看 `roi-preview` 截图：红框内应能看到播报条，框外尽量不要包含大片高亮的天空/爆炸特效。
2. 看 `outputs/<名>/signals.csv` 的 `ink_rise` 列：真实击杀播报出现时该列应有明显尖峰（>0.08），平静时段接近 0。若整列都很平，说明 ROI 没框对或阈值太高；若持续非零，说明框进了高亮背景。
3. 播报条**淡出**不算事件：检测器只认「HUD 高亮像素占比上升」，这是为了避免把一次击杀算成两次。

> 游戏版本更新会挪动 HUD。建议把每版 UI 对应的 ROI 存成一份 `config-2k.json` / `config-zh.json`，用 `--config` 切换，别改代码。

---

## 6. 配置文件全表（`config.json`）

所有参数都可被命令行覆盖。改 `config.json` 是「长期设定」，命令行参数是「这一次试试」。

### `roi` — 检测区域

| 字段 | 默认值 | 说明 |
|---|---|---|
| `x` / `y` / `w` / `h` | 0.72 / 0.05 / 0.27 / 0.22 | 比例坐标，见第 5 节 |

### `sample` — 抽帧

| 字段 | 默认值 | 说明 |
|---|---|---|
| `fps` | 8 | 每秒采样帧数。越低越快、越容易漏短事件；真实录像是 60fps，8fps 足够抓播报条 |
| `width` / `height` | 96 / 32 | ROI 缩放后的灰度小图尺寸。越大越准但越慢，一般不用改 |

### `recognizer` — 神经网络识别后端（P1 新增）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `backend` | none | `none` = 启发式（零依赖）；`crispy` = Crispy 预训练击杀检测（见第 16 节） |
| `game` | overwatch | 游戏模型：`overwatch` / `csgo2` / `valorant`，对应各自的 HUD 检测框与权重 |
| `weights_dir` | models/crispy | 权重目录（相对项目根）。`python run.py fetch-model` 会下载到这里 |
| `confidence` | 0.6 | 击杀概率阈值。漏检 → 0.4~0.5；误检 → 0.7~0.8（Crispy 官方 release 默认 0.8） |
| `fps` | 8 | 模型采样帧率（独立于 `sample.fps`） |
| `stretch` | "auto" | 归一化模式：`auto` 按视频宽高比自动选（4:3 → 1440×1080 均匀缩放，16:9 → 1920×1080）；`true`/`false` 手动覆盖。**4:3 录像（如 1280×960）必须均匀缩放**，否则 HUD 图标横向变形导致模型漏检 |

### `detect` — 视觉事件判定（调参主战场）

| 字段 | 默认值 | 说明 / 调参方向 |
|---|---|---|
| `diff_gain` | 60.0 | 帧差归一化除数。画面抖动大 → 调大 |
| `rise_gain` | 0.25 | HUD 新增面积归一化除数。播报条很小 → 调小 |
| `ink_gain` | 0.45 | 绝对 HUD 面积归一化除数 |
| `w_diff` / `w_rise` / `w_ink` | 0.45 / 0.40 / 0.15 | 三路信号权重，和应为 1 |
| `threshold` | 0.55 | **最常用**：判定阈值。漏检多 → 降到 0.35~0.45；误检多 → 升到 0.65~0.75 |
| `min_rise` | 0.08 | HUD 面积最小增量，用来排除背景运动与播报淡出 |
| `refractory_sec` | 0.6 | 两次事件最小间隔（冷却）。连杀被吞 → 调小到 0.4 |

### `audio` — 音频峰值

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enabled` | true | 关掉用 `--no-audio` |
| `sample_rate` | 8000 | 单声道 PCM 采样率 |
| `window_sec` | 0.10 | RMS 统计窗口 |
| `peak_ratio` | 2.6 | 相对 EMA 基线多少倍算峰值。语音/解说多 → 调大 |
| `refractory_sec` | 0.5 | 峰值冷却 |
| `baseline_alpha` | 0.05 | 基线更新速度 |
| `min_rms` | 0.012 | 绝对音量下限，过滤静音段噪声 |

### `clip` — 片段切分

| 字段 | 默认值 | 说明 |
|---|---|---|
| `pre_roll` | 4.0 | 事件前留白（铺垫）。参考 Crispy 对 OW2 的推荐值 |
| `post_roll` | 3.0 | 事件后留白（收尾） |
| `merge_gap` | 5.0 | 相邻事件间隔小于它就归并成一段（连杀） |
| `min_len` | 3.0 | 片段最短时长 |
| `max_len` | 16.0 | 片段最长时长，超了保留高潮、裁掉开头 |

> **参数耦合提醒**：`merge_gap` 若明显小于 `pre_roll + post_roll`，相邻片段会重叠。代码已自动处理（重叠 ≤2s 裁掉后段起点，>2s 合并成一段），但想要「多段独立集锦」建议让 `merge_gap ≥ pre_roll + post_roll`。

### `select` — 选段

| 字段 | 默认值 | 说明 |
|---|---|---|
| `top_k` | 6 | 最多入选片段数 |
| `max_total_sec` | 120.0 | 成片总时长上限 |
| `min_score` | 0.0 | 得分下限，用来砍掉低质量候选 |
| `playerless_penalty` | 0.15 | 玩家中心性降权：全片确认过玩家击杀时，无玩家击杀的段扣分；0=关闭 |

### `render` — 输出

| 字段 | 默认值 | 说明 |
|---|---|---|
| `width` / `height` | 0 / 0 | 横版输出分辨率；**0 = 跟随源**（1080p 源出 1080p 成片，无降采样、无 letterbox）。给正数则按指定分辨率缩放（保持原比例 letterbox） |
| `fps` | 0 | 输出帧率；**0 = 跟随源**（60fps 录像保持 60fps，甩镜头不顿挫） |
| `crf` | 18 | H.264 质量（18 接近视觉无损；23 是 x264 默认档，枪战运动画面会有可见块效应） |
| `preset` | medium | 编码速度档（画质优先；嫌慢可回 `fast`/`veryfast`） |
| `vertical` | false | true 输出 9:16 竖版 1080×1920（居中裁切） |
| `burn_subtitles` | false | 是否把 SRT 烧进画面（需 libass，失败会自动跳过）。开启时中间切片用近无损档（crf 10）编码，最终画质只由烧录阶段编码一次决定——**全程仅一代有损压缩** |
| `bgm` | null | BGM 文件路径 |
| `bgm_volume` | 0.18 | BGM 混音音量 |

> **画质说明（v0.7.0）**：旧默认（1280×720@30fps、crf 23、veryfast）会把 1080p60 录像降采样到
> 56% 像素、帧率砍半、再叠两层有损编码，成片画质明显糊于源。新默认跟随源分辨率/帧率 +
> crf 18，实测 1080p60 成片码率约 10 Mbps，与源逐帧对比仅轻微差异、HUD 文字锐度一致。

### `agent` — 编排层

| 字段 | 默认值 | 说明 |
|---|---|---|
| `llm_enabled` | false | 是否调用 LLM 生成解说词 |
| `api_key_env` | `OWAGENT_API_KEY` | 读取 API Key 的环境变量名 |
| `base_url` | `https://api.openai.com/v1` | 任意 OpenAI 兼容端点 |
| `model` | `gpt-4o-mini` | 模型名 |
| `timeout_sec` | 15 | 超时后自动降级为本地文案 |

启用示例（**不联网/没 Key 也不会报错，会自动降级**）：

```bash
# Windows PowerShell
$env:OWAGENT_API_KEY="sk-xxx"
python run.py process "D:/videos/my_match.mp4" --llm

# macOS / Linux / Git Bash
OWAGENT_API_KEY=sk-xxx python run.py process ./my_match.mp4 --llm
```

---

## 7. 命令行参考

```bash
python run.py <子命令> [视频路径] [参数]
```

| 子命令 | 作用 |
|---|---|
| `demo` | 生成合成录像并跑通全流程 + 真值准确率对比 |
| `self-test` | 只生成合成录像与真值 JSON（`--out-dir` 指定目录） |
| `probe <video>` | 打印分辨率、时长、音轨、当前 ROI 像素框、识别后端状态 |
| `roi-preview <video>` | 导出带 ROI 框的 PNG，用于标定 |
| `process <video>` | 完整处理一段录像 |
| `fetch-model` | 下载 Crispy 预训练权重（约 21 MB，一次性；`--game` 选游戏或 `all`） |

`process` / `demo` 可用参数（`probe` / `roi-preview` 也接受配置类参数）：

| 参数 | 作用 |
|---|---|
| `--config PATH` | 指定配置文件（默认 `./config.json`） |
| `--roi x,y,w,h` | 覆盖 ROI 比例 |
| `--threshold F` | 覆盖事件判定阈值 |
| `--fps F` | 覆盖采样帧率 |
| `--top-k N` | 覆盖最多入选段数 |
| `--pre F` / `--post F` / `--gap F` | 覆盖前留白 / 后留白 / 合并间隔（秒） |
| `--no-audio` | 禁用音频通道 |
| `--recognizer none\|crispy` | 切换识别后端（默认 none；见第 16 节） |
| `--game overwatch\|csgo2\|valorant` | crispy 后端的游戏模型 |
| `--confidence F` | crispy 后端的击杀概率阈值 |
| `--vertical` | 输出 9:16 竖版 |
| `--burn` | 烧录字幕 |
| `--bgm PATH` | 混入背景音乐 |
| `--llm` | 启用 LLM 解说词 |
| `--dry-run` | 只检测与决策，不渲染视频 |
| `--workdir DIR` | 产物目录（默认 `outputs/<视频名>`） |
| `--out PATH` | 成片路径（默认 `<workdir>/highlights.mp4`） |
| `--time F` | 仅 `roi-preview`：截帧时间点，默认 3.0s |
| `--png PATH` | 仅 `roi-preview`：截图输出路径 |

几个常用组合：

```bash
# 只调参、不花时间编码
python run.py process match.mp4 --threshold 0.4 --gap 7 --dry-run

# 神经网络识别（先 fetch-model 下载权重）
python run.py process match.mp4 --recognizer crispy --confidence 0.6 --dry-run

# 抖音/短视频竖版 + 字幕 + BGM
python run.py process match.mp4 --vertical --burn --bgm ./bgm.mp3 --top-k 5

# 用另一套 UI 配置处理 2K 录像
python run.py process match_2k.mp4 --config config-2k.json --roi 0.74,0.04,0.25,0.20
```

---

## 8. 产物说明

默认输出到 `outputs/<视频名>/`：

| 文件 | 内容 |
|---|---|
| `highlights.mp4` | 成片（或用 `--out` 指定的位置） |
| `timeline.json` | **Agent 的状态快照**：全部候选、入选标记、事件时间戳、得分、理由、解说词 |
| `signals.csv` | 每采样帧的 `t / diff / ink / ink_rise / level / score`，**调阈值就看它** |
| `audio_rms.csv` | 音频能量包络 `t / rms` |
| `captions.srt` | 按成片时序排好的字幕，可直接拖进播放器或剪映/PR |
| `report.md` | 中文决策报告：入选表、落选候选、打分公式、参数快照、自动调参建议、产物清单 |
| `parts/` | 各切片中间文件与 concat 列表，排查用 |

---

## 9. 工作原理

```
录像 mp4
  │
  ├─ ffmpeg: crop(ROI) → fps=8 → scale 96x32 → gray → rawvideo(stdout)
  │    Python 标准库逐帧算 diff / ink / rise → 上升沿 + 冷却 → 视觉事件
  │
  ├─ ffmpeg: -ac 1 -ar 8000 -f s16le → PCM(stdout)
  │    Python array 算 0.1s 窗 RMS → 相对 EMA 基线峰值 → 音频事件
  │
  ├─ 合并两路事件 → 按 merge_gap 聚类 → 生成 [first-pre, last+post] 片段
  │    打分 → Top-K → 按源片时序重排 → 重叠消解 → timeline.json
  │
  ├─ Agent 编排层（只消费 timeline.json，不读任何帧）
  │    决策说明 / 可选 LLM 解说 / 自动调参建议 → report.md
  │
  └─ ffmpeg 逐段转码成统一参数 → concat → 可选字幕烧录 / BGM → highlights.mp4
```

**视觉信号**（每帧 `gray` 小图，`HIGHLIGHT_PIXEL = 170`）

- `diff`：与上一帧的平均绝对差（0~255）
- `ink`：灰度 > 170 的像素占比（HUD 白字与图标）
- `rise = max(0, ink − 上一帧 ink)`：**新出现的 HUD 面积**
- `score = w_diff·min(diff/diff_gain,1) + w_rise·min(rise/rise_gain,1) + w_ink·min(ink/ink_gain,1)`
- 判定：`rise ≥ min_rise` **且** `score ≥ threshold` **且** 距上次事件 ≥ `refractory_sec`

要求 `rise` 是关键设计：播报条淡出、持续显示、镜头甩动造成的背景变化都不会产生新事件，从根上避免了「一次击杀算两次」。

**打分公式**

```
score = 0.40 × 峰值强度
      + 0.30 × 连击度        # min(同窗视觉事件数 / 4, 1)
      + 0.20 × 视听交叉确认   # 同窗内同时有视觉与音频事件 = 1
      + 0.10 × 上下文        # 避开片头片尾 5s = 1，否则 0.6
```

选段后**按源片时间重排**（不是按分数），集锦叙事才连贯；相邻片段重叠时，≤2s 裁掉后段开头，>2s 合并成一段连续高光。

**渲染**：每段单独用 FFmpeg 转码到统一分辨率/帧率/编码/音轨参数（无音轨自动补静音），再用 concat demuxer 无损拼接，因此不会出现时间戳错乱。字幕烧录和 BGM 是拼接后的可选后处理，失败不影响成片。

---

## 10. 调参指南

先看 `signals.csv`，再看 `report.md` 末尾的自动建议。

| 现象 | 先看什么 | 怎么调 |
|---|---|---|
| 一个事件都检不到 | `ink` / `ink_rise` 是否全 0 | 重标 ROI（`roi-preview`）；`--threshold 0.30` 试水 |
| 漏检多（真高光没进） | `ink_rise` 有峰但 `score` 没过线 | 降 `--threshold`；调小 `rise_gain`；调小 `refractory_sec` |
| 误检多（一堆废片段） | 平静时段 `diff` 是否很大 | 升 `--threshold`；调大 `diff_gain`；调大 `min_rise`；缩小 ROI 避开特效区 |
| 连杀被算成一次 | 事件间隔与 `refractory_sec` | 调小 `refractory_sec`（0.4）；或把 `merge_gap` 调大让它们归成一段 |
| 片段太碎 | `timeline.json` 的候选数 | 调大 `--gap`（合并间隔）、调小 `--top-k` |
| 片段过长 / 全是铺垫 | 单段 duration | 调小 `--pre` / `--post`；调小 `clip.max_len` |
| 成片画面重复 | 相邻片段 start/end | 代码已自动消解；仍出现就调大 `--gap` |
| 音频峰值太多/太少 | `audio_rms.csv` | 调 `audio.peak_ratio`；或 `--no-audio` 先只靠视觉 |

**推荐顺序**：先 `--dry-run` 反复调 `threshold` 和 `roi` → 确认 `timeline.json` 里的事件时间戳和真实高光对得上 → 再正式渲染。dry-run 不编码，几秒就能跑一轮。

---

## 11. 常见问题

**Q：提示「找不到 ffmpeg」？**
确认 `ffmpeg -version` 能跑通。Windows 用 winget 装完要**重开终端**刷新 PATH。也可以把 `ffmpeg.exe` 放进项目目录后把该目录加入 PATH。

**Q：路径有中文/空格会挂吗？**
不会，所有命令都用参数数组传路径，不用 shell 字符串拼接。命令行里给路径加引号即可。

**Q：处理一小时的录像要多久？**
检测是纯 CPU 的轻量逐帧运算（8fps × 3600s = 28800 帧 × 3072 字节），通常几十秒；瓶颈在 FFmpeg 逐段转码。想加速：降 `render.preset` 到 `ultrafast`、降输出分辨率、减小 `--top-k`。

**Q：4K / 21:9 / 竖屏录像支持吗？**
支持任意分辨率，ROI 用比例坐标自动换算。21:9 的 HUD 位置可能偏中间，需要重新标定 ROI。

**Q：烧录的中文字幕是方框？**
`libass` 找不到中文字体。装系统中文字体（Windows 自带微软雅黑通常没问题），或改用 `--burn` 关闭烧录、直接加载 `.srt` 到剪辑软件。

**Q：LLM 一定要联网吗？**
不必须。不加 `--llm` 时全程离线；加了但没 Key / 网络失败，会自动降级为本地规则文案并打印警告，不会中断流程。

**Q：能不能直接读游戏回放（replay）文件？**
不能，也不建议。暴雪没有开放通用的回放结构化接口，本原型只处理**录屏视频的像素与音轨**。用 OBS 录下来再喂给它即可。

**Q：会不会被反作弊判定为外挂？**
不会。程序不注入游戏进程、不读内存、不抓包、不修改游戏文件，只是一个离线视频分析脚本。但请注意只处理**自己录制的素材**，公开分享时留意游戏内容版权与队友隐私。

**Q：支持 mp4 以外的格式吗？**
FFmpeg 能解的基本都行（mkv / mov / ts / flv…），输出统一 mp4。

---

## 12. 升级到 P1 / P2 的接入点

代码按层拆开，每一层都能单独替换，不用重写整条链路：

| 层 | 文件 | 现状 | 升级方向 |
|---|---|---|---|
| 视觉召回 | `owagent/detector.py` | ROI 像素信号（启发式） | 接 PaddleOCR/Tesseract 读播报文字；接 YOLO 认英雄图标；接微调 X-CLIP / VLM 对候选窗口做语义复核 |
| 模型召回 | `owagent/model.py` | **Crispy 预训练 MLP 击杀检测（已接入，第 16 节）** | 换更强的时序模型（X-CLIP）；用自己录像的标注微调；多游戏混合召回 |
| 音频召回 | `owagent/audio.py` | RMS 峰值 | 接 faster-whisper 转写大招语音；按英雄语音线做模板匹配 |
| 打分 | `owagent/timeline.py` | 加权公式 | 用人工反馈数据训练 LightGBM 学习排序；加残血、加时、占点等特征 |
| 编排 | `owagent/agent.py` | 规则报告 + 可选 LLM | 接真实工具调用（发布、通知）、人在回路 UI、长期偏好记忆 |
| 渲染 | `owagent/renderer.py` | 硬切拼接 | 转场、慢放回放、动态裁剪跟随准星、自动卡点 BGM |
| 采集 | `owagent/cli.py` | 本地文件 | 接 OBS 环形缓冲做准实时；批量目录监听 |

**可直接参考的现成项目**
- [Crispy](https://github.com/Flowtter/crispy)：OW2 / CS2 / 瓦罗兰特的神经网络高光检测——**本项目的模型召回层（`owagent/model.py`）已接入其预训练权重**，官方推荐参数也是本原型 `clip` 段的取值来源（confidence 0.6 / 8fps / 前 4s / 后 3s / 击杀间隔 5s）
- [OCRwatch](https://github.com/InventivetalentDev/OCRwatch)：Tesseract 读 OW2 HUD 的参考实现
- [Gameplay Highlights Generation](https://arxiv.org/abs/2505.07721)（AMD, 2025）：微调 X-CLIP 跨游戏泛化、ONNX + DirectML 在 Windows 部署；论文也指出纯 OCR 路线「每款游戏都要单独工程、跨语言泛化差」，这正是后续要转向 VLM 的理由

**建议节奏**：P0 用规则跑通闭环 → P1 用 VLM 提质 → P2 才把 LLM 放进编排层。反过来做（一上来让大模型逐帧看视频）会又贵又不准。

---

## 13. 项目结构

```
ow-highlight-agent/
├── run.py                    # 命令行入口
├── gui.py                    # 桌面图形界面（tkinter，零额外依赖）
├── app.py                    # Streamlit 图形界面入口（浏览器版）
├── test_gui_smoke.py         # 桌面界面冒烟测试（隐藏窗口自动回归）
├── requirements.txt          # 图形界面依赖：streamlit
├── requirements-model.txt    # 神经网络识别后端依赖：numpy (+ pillow)
├── requirements-ocr.txt      # OCR 击杀播报读取依赖：rapidocr + onnxruntime + opencv
├── start_gui.bat             # Windows 一键启动桌面窗口（推荐）
├── start_ui.bat              # 等价于 start_gui.bat（旧习惯入口）
├── .streamlit/config.toml    # 上传上限 / 端口 / 浅色主题
├── config.json               # 全部可调参数（也是界面控件默认值）
├── README.md                 # 本文档
├── owagent/
│   ├── __init__.py
│   ├── util.py               # ffmpeg 定位、命令执行、媒体探测、日志
│   ├── config.py             # 配置加载/校验/命令行覆盖
│   ├── selftest.py           # 合成测试录像 + 真值生成 + 准确率对比
│   ├── detector.py           # L1 启发式视觉召回（ROI 灰度信号）
│   ├── model.py              # L1 模型召回：Crispy 预训练击杀检测 + 权重下载
│   ├── reviewer.py           # L2 语义复核：DeepSeek VLM 看关键帧判定事件/精彩度/解说
│   ├── audio.py              # L1 音频 RMS 峰值召回
│   ├── timeline.py           # 片段聚类、打分、Top-K、重叠消解
│   ├── renderer.py           # FFmpeg 切片/拼接/字幕/BGM
│   ├── agent.py              # 编排层：决策报告 + 可选 LLM 解说
│   ├── cli.py                # 命令行子命令与流程编排
│   └── webui.py              # 界面桥接层（不依赖 streamlit，可单独调用）
├── models/crispy/            # 预训练权重目录（.npy 不入 git，fetch-model 下载）
│   ├── SOURCE.md             # 模型来源、MIT 许可与致谢说明
│   ├── csgo2-mask.png        # CS2 预处理遮罩
│   └── valorant-mask.png     # 瓦罗兰特预处理遮罩
├── samples/                  # demo 生成的合成测试录像与真值
├── uploads/                  # 界面上传的临时录像
└── outputs/                  # 所有产物
```

---

## 14. 合规说明

- 本工具是**离线录像分析脚本**，不读取游戏内存、不注入进程、不拦截网络、不修改游戏客户端。
- 仅建议处理你本人录制、有权使用的画面；公开分享前留意游戏内容版权与队友隐私。
- 合成测试录像由代码生成，不含任何游戏素材，可放心用于演示与二次开发。
- 预训练权重来自 MIT 许可的 [Crispy](https://github.com/Flowtter/crispy) 项目（见 `models/crispy/SOURCE.md`），仅用于本地录像分析。

---

## 15. 图形界面（Streamlit 本地客户端）

> **更省事的替代**：第 17 节的**桌面窗口版**（tkinter）零额外依赖、双击即开、无需起服务和管理端口，日常使用推荐它；本节的 Streamlit 版保留给喜欢浏览器操作与页面内曲线图的场景。

不想记命令的话，用 `app.py` 可以在浏览器里完成全部操作。需要 `pip install -r requirements.txt`；检测与剪辑逻辑仍复用同一套 `owagent/` 模块，界面不重复实现任何算法。

### 启动

```bash
cd ow-highlight-agent
pip install -r requirements.txt
python -m streamlit run app.py
```

Windows 也可以直接双击 `start_ui.bat`——注意它现在默认打开**桌面窗口版**（见第 17 节）；要用浏览器版请手动执行上面的 `streamlit run` 命令。启动后请**保持黑窗口开着**，关掉窗口等于关掉服务。

指定端口或不想自动开浏览器：

```bash
python -m streamlit run app.py --server.port 8700 --server.headless true
```

### 界面功能

| 区域 | 能做的事 |
|---|---|
| 视频来源 | 本地绝对路径 / 上传文件 / 一键生成合成 Demo |
| 左侧参数栏 | ROI 四个比例、判定阈值、采样帧率、**识别后端（启发式 / Crispy 神经网络 + 游戏模型 + 击杀概率阈值 + 一键下载权重）**、前后留白、合并间隔、Top-K、竖版、烧字幕、BGM、LLM、dry-run、产物目录 |
| 辅助按钮 | 「探测视频」看分辨率/时长/音轨；「预览 ROI 框」直接显示带红框的截帧 |
| 运行过程 | 「开始处理」后实时滚动显示 `[probe] / [model] / [detect] / [timeline] / [render] / [agent]` 日志 |
| 结果区 | 四项指标卡（模型模式显示「模型击杀」数）、页面内播放成片、入选片段表格、逐帧 `score / ink_rise / p_kill / 阈值` 曲线、真值召回率对照 |
| 下载 | 一键下载 MP4、SRT、timeline.json、决策报告；报告可直接在页面展开阅读 |

### 推荐操作顺序

1. 选「合成 Demo（一键体验）」→ 点「开始处理」，先确认整条链路能出片（界面会显示召回率 / 精确率）。
2. 换成自己的录像，点「探测视频」确认能读到，再点「预览 ROI 框」看红框是否盖住右上角击杀播报。
3. ROI 不对就调左侧四个比例；勾上「只检测不渲染」快速试阈值，看信号曲线里 `ink_rise` 的尖峰。
4. 取消 dry-run，点「开始处理」出片，预览满意后下载成片与字幕。

### 实现说明

- `app.py` 只负责渲染控件；`owagent/webui.py` 是无 Streamlit 依赖的桥接层，负责把控件值转成配置、调用 `process_video`、把日志通过回调转发给界面。因此这段逻辑可以脱离界面单独用 Python 调用与测试。
- 侧栏控件的初始值来自 `config.json`，页面上的改动**只影响本次运行**，不会写回配置文件；想固化就去改 `config.json`。
- 结果保存在 Streamlit 的 session state 中，调整侧栏参数不会丢失；点「清空结果」可重置。
- `.streamlit/config.toml` 已把上传上限设为 2048 MB、主题设为浅色、关闭用量统计。即便如此，**数 GB 的录像仍建议用「本地路径」**，不要走浏览器上传。
- Streamlit 在本机起一个 HTTP 服务，视频文件全程留在本地，不会上传到任何外部服务。
- 默认端口是 **8600**（不是 Streamlit 默认的 8501），用于避开可能已被占用的 8501；改端口请编辑 `.streamlit/config.toml` 的 `server.port`，或启动时加 `--server.port`。

### 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: streamlit` | 执行 `pip install -r requirements.txt`；Windows 可试 `py -3 -m pip install -r requirements.txt`，或让 `start_ui.bat` 自动挑解释器 |
| 双击 `start_ui.bat` 打不开页面 | 黑窗口若显示 `[ERROR] No Python interpreter with Streamlit`，按提示装依赖；若显示 `Port 8600 is not available`，说明服务已在跑，直接访问 <http://localhost:8600> 即可。脚本本身已能自动处理这两种情况 |
| 端口 8600 被占用 | 换端口：`--server.port 8700`，或改 `.streamlit/config.toml` |
| 上传大文件失败 | 改用「本地路径」；`maxUploadSize` 已在配置里调到 2048 MB |
| 成片预览加载不出来 | 文件太大时浏览器解码慢，用「成片 MP4」下载按钮或直接打开 `outputs/<名>/highlights.mp4` |
| 改了参数没生效 | 参数在点击「开始处理」时才读取，改完要重新点一次 |
| 界面报错但命令行正常 | 展开运行结果的「运行日志」与错误详情；也可先用 `python run.py demo` 确认底层无问题 |

---

## 16. 神经网络识别后端（Crispy 预训练模型）

P0 的启发式只认「HUD 亮条出现」，不认识画面语义。本节介绍可选的**语义级击杀召回**：接入 [Crispy](https://github.com/Flowtter/crispy)（MIT 许可）发布的预训练神经网络——它就是用真实《守望先锋》等游戏的击杀画面训练的轻量 MLP（10000→120→15→2），每帧输出一个「此刻画面里有没有击杀播报」的概率。

**为什么选它**：任务完全对口（FPS 击杀画面检测）、纯 numpy 推理**不需要 PyTorch/ONNX 等重型框架**、权重只有 9.4 MB、支持守望先锋 / CS2 / 瓦罗兰特、MIT 许可可自由集成。

### 16.1 启用三步

```bash
# ① 装可选依赖（numpy 必需；CS2/瓦罗兰特另需 pillow）
pip install -r requirements-model.txt

# ② 下载预训练权重（约 21 MB，一次性；不进 git）
python run.py fetch-model              # 只要守望先锋
python run.py fetch-model --game all   # 三个游戏都要

# ③ 用模型跑真实录像（先 dry-run 看召回）
python run.py process "D:/videos/match.mp4" --recognizer crispy --dry-run
python run.py process "D:/videos/match.mp4" --recognizer crispy
```

界面里等价操作：侧栏「神经网络识别」→ 选「Crispy 神经网络」→ 没有权重时点「下载模型权重」→ 开始处理。

### 16.2 它是怎么工作的

| 游戏模型 | HUD 检测框（1920×1080 基准） | 预处理 | 权重 |
|---|---|---|---|
| overwatch | 画面中心 (910, 490) 100×100 | 红色通道增强（OW2 击杀确认标记为红/白）→ 灰度 | `overwatch.npy` |
| csgo2 | 中下 (930, 925) 100×100 | 边缘检测 + 遮罩 | `csgo2.npy` |
| valorant | 中下 (899, 801) 122×62 | 边缘检测 + 遮罩拼接 | `valorant.npy` |

流程：ffmpeg 先把任意分辨率的帧**按 `stretch` 配置归一**（默认 `auto`：按视频宽高比自动选——4:3 录像均匀缩放到 1440×1080，16:9 录像缩放到 1920×1080；也可手动 `true`/`false` 覆盖），裁出对应游戏的 HUD 框 → 逐帧预处理 → MLP 输出 `p_kill` → **`p_kill` 越过 `confidence` 的上升沿记为一次击杀事件**（`source: "model"`）→ 与启发式 kill-feed 召回**双通道合并去重**（1.5s 内视为同一击杀，两通道互为兜底）→ 进入聚类、打分、剪辑流水线。`signals.csv` 在此模式下记录每帧 `t / p_kill`，界面曲线图会画出概率与阈值线。

> **宽高比陷阱（v0.3.0 修复）**：4:3 录像（如 CS2 常见的 1280×960）若被简单拉伸到 1920×1080，HUD 图标会横向变形约 33%，MLP 直接认不出——实测三杀瞬间 `p_kill` 从 0.98 跌到 0.01。`stretch: "auto"` 会按宽高比自动选择均匀缩放，此问题不再需要用户手动处理。

### 16.3 参数与调参

| 现象 | 处理 |
|---|---|
| 整场 `p_kill` 都贴着 0 | 画面不是对应游戏的全屏画面；确认录像无黑边、无窗口边框、游戏模型选对；4:3 录像确认 `stretch` 为 `auto`（v0.3.0 起默认） |
| 击杀漏检 | `--confidence` 降到 0.4~0.5；v0.3.0 起模型漏检时启发式 kill-feed 通道会兜底召回 |
| 误检偏多 | `--confidence` 升到 0.7~0.8（Crispy 官方 release 默认 0.8） |
| 连杀只报一次 | 正常：`clip.merge_gap`（默认 5s）内的相邻击杀会合并成一段连杀集锦 |
| 提示「找不到权重文件」 | 运行 `python run.py fetch-model`；或检查 `recognizer.weights_dir` |
| numpy / pillow 未安装 | `pip install -r requirements-model.txt`；**权重缺失或依赖缺失都不会中断出片**，会打印警告并自动回退启发式检测 |

### 16.4 本仓库的验证口径

- **预处理等价性**：向量化实现与 Crispy 原版逐像素循环在随机帧上对比，**逐像素差值 = 0**（完全一致，见 `_preprocess_overwatch`）。
- **正对照**：在 OW2 检测框内合成 OW2 风格红色击杀标记，`p_kill` 从 0.01 升至 **0.99**；小标记（30px）不触发，与 1080p 下真实标记尺寸一致。
- **负对照**：真实 CS2 录像（无击杀发生的 2.5s 片段）全程 `p_kill ≈ 0.01`，无误报。
- **回退**：权重/依赖缺失时自动回退启发式并给出明确提示，已实测。

> 局限：MLP 只回答「有没有击杀播报」，不区分多杀/大招/团灭；HUD 版本大改后分布漂移需要重新训练（训练脚本在上游 Crispy 仓库）。这层负责「召回」，语义分级交给后续的 VLM 复核层。

### 16.5 许可与致谢

权重与遮罩文件来自 Crispy release 1.1.4（MIT License），来源、下载方式与致谢见 `models/crispy/SOURCE.md`。本仓库仅复刻其推理管线（`owagent/model.py`），训练数据与权重版权归 Crispy 项目所有。

---

## 17. 桌面图形界面（tkinter，推荐日常使用）

第 15 节的 Streamlit 版需要起本地 HTTP 服务、装依赖、管理端口，出过「双击打不开」的问题。本节是一个**原生桌面窗口版**（`gui.py`）：

- **零额外依赖**：tkinter 是 Python 自带标准库，不用 `pip install` 任何界面框架；
- **双击即开**：没有服务、没有端口、没有浏览器；关窗口即退出，不存在残留进程；
- **逻辑零重复**：与 CLI、Streamlit 版共用同一套 `owagent/` 模块与 `config.json`，界面只做参数收集与日志展示。

### 17.1 启动

Windows 直接双击 **`start_gui.bat`**（或 `start_ui.bat`，两者等价）。脚本会自动挑选一个**同时装有 tkinter 和 numpy** 的 Python 解释器（tkinter 随 python.org 官方安装包自带；numpy 若缺按提示 `py -3 -m pip install numpy`），然后无控制台启动窗口。

其它系统或手动启动：

```bash
cd ow-highlight-agent
python gui.py        # 需要该解释器带 tkinter（标准安装自带）和 numpy
```

### 17.2 界面布局与功能

| 区域 | 能做的事 |
|---|---|
| 1 · 录像文件 | 「浏览…」选文件（自动探测分辨率/时长/音轨）；「ROI 预览」按指定秒数截帧并弹窗显示带红框/黄框的截图，用于确认 ROI 盖住击杀播报区 |
| 2 · 识别后端 | 启发式信号 / Crispy 神经网络二选一；选后者后可选游戏（OW/CS2/瓦罗兰特）、拖动击杀置信度；「下载模型权重」一键拉取（约 21 MB，一次性），下载完成状态灯即时刷新 |
| 3 · 参数 | 判定阈值、最多片段数、事件前后留白；勾选「只检测不出片 (dry-run)」「竖版 9:16」「烧录字幕」 |
| 4 · 环境状态 | ffmpeg / ffprobe / numpy / 当前游戏权重四项自检，异常时整行变红 |
| 运行日志 | 黑底彩色滚动日志（`[probe]/[model]/[detect]/[timeline]/[render]/[agent]`），超 8000 行自动截断 |
| 入选高光片段 | 表格列出每段的序号、起止时间、得分、判定依据；一段没有时给出调参提示 |
| 操作条 | 「▶ 开始处理」「🧪 生成演示并跑通全流程」（合成 Demo 一键体验）；完成后「📂 打开产物文件夹」「🎬 播放成片」 |

推荐顺序：先点「生成演示」确认链路通 → 选自己的录像 → 「ROI 预览」确认框位 → 勾 dry-run 试阈值 → 正式出片。

### 17.3 实现说明与故障排查

- 流水线在**后台 daemon 线程**运行，界面不卡死；工作线程只写队列，主线程每 100ms 消费队列更新日志与界面（唯一的 UI 更新入口，无跨线程 Tk 调用）。
- 处理期间按钮置灰防误触；关窗口时若仍在跑会弹确认框（强退会中断 ffmpeg，留下半成品但不损坏源文件）。
- 双击 bat 后窗口没出现：到项目目录看是否有 `error.log`（启动期异常会写到那里）；确认装了 numpy（`py -3 -m pip install numpy`）后重试。
- `python gui.py` 报 `No module named tkinter`：说明当前解释器是精简版（如某些嵌入式发行版），换 python.org 官方安装包自带的解释器，或用 `start_gui.bat` 自动挑一个。
- 冒烟测试：`python test_gui_smoke.py` 会隐藏窗口自动跑参数链路 + demo 全流程 + 结果/日志断言，改完 `gui.py` 后可一键回归。
- **窗口大小自适应（v0.7.0）**：窗口尺寸在布局完成后按控件实际需求自动计算（不再写死 1200×820），`minsize` 也取需求值——无论怎么拖都不会裁掉内容；底部操作条最先 pack 到底部，屏幕实在装不下时压缩的是可滚动的日志区，按钮行永远可见。启动时声明 Windows DPI 感知并按缩放系数放大字体（高分屏文字不再发虚、布局计算不再因逻辑分辨率失真）；左栏参数区做了双列紧凑排布（游戏+置信度同行、OCR+游戏ID 同行、参数 5 行并 3 行），125% 缩放的 1080p 屏也能完整显示。

---

## 18. L2 语义复核（DeepSeek VLM 大模型）

第 16 节的 Crispy MLP 只会回答「这一帧有没有击杀播报」——分不清单杀和五杀、击杀和死亡回放、也不懂什么叫精彩。本节接入**视觉语言大模型**做语义复核：VLM 亲眼看入选片段的关键帧，回答「发生了什么、多精彩」，并把结果融合进最终排序。

**三游戏全支持**：复核提示词、事件词表、融合权重均按 `--game` 自动切换（见 18.6）。

### 18.1 工作原理（漏斗架构，成本可控）

```
整场录像 ──► L1 召回（Crispy/启发式，全帧扫描，廉价）
              └─► 候选片段 Top-K（默认 ≤ 8 段）
                    └─► L2 语义复核（本节，每段抽 3~4 张关键帧，一次 VLM 调用）
                          ├─ 事件类型：按游戏分词表（OW：团灭/连杀/大招…；CS2/VAL：五杀ACE/残局/拆弹…）
                          ├─ VLM 精彩分 0-10 + 连杀数 + 置信度
                          └─ 一句话中文解说（直接进字幕和报告）
                                └─► L3 融合打分：w_signal×信号分 + w_vlm×精彩分 + w_event×事件权重
```

一场 5 分钟录像实测：**34 次信号召回 → 6 段复核 → 69 秒跑完全链路，消耗约 1.8 万 tokens**（几分钱）。VLM 的价值实测示例：

- 源氏开大双杀 → `连杀×2, 精彩 7/10`「源氏开大连续击杀两名敌人」，综合分冲到 0.844 排第一
- Crispy 报 p=0.99 的「击杀」→ VLM 识破是**死亡回放**，精彩分压到 2/10（MLP 无法区分的误检被语义层拦下）
- 「死神贴脸输出拿下一杀」——连英雄都认得

### 18.2 启用步骤

```bash
# ① 配置 API Key（三选一）
#    a) 桌面 GUI 的「2b · 语义复核」区直接输入（自动保存，推荐）
#    b) 写入 config.local.json（已被 .gitignore 排除）：
{"review_api_key": "sk-你的deepseek密钥"}
#    c) 或环境变量：set OWAGENT_REVIEW_KEY=sk-...

# ② 命令行启用
python run.py process "D:/videos/xxx.mp4" --recognizer crispy --review
python run.py process "D:/videos/xxx.mp4" --recognizer crispy --review --dry-run   # 先看复核结果不出片
```

桌面 GUI 中：勾选「启用 DeepSeek 视觉复核」后，在 **API Key 输入框**粘贴 Key（默认打码显示，可勾「显示」查看）；「测试连接」按钮随时手动验证可用性。**每次开始处理前会自动做一次连通性检测**（调 `/models`，不消耗 tokens）——检测失败时弹窗询问：继续（自动跳过复核）或中止。

桌面 GUI：勾选「2b · 语义复核」即可（Key 状态自检显示在勾选框下方）。

### 18.3 默认模型与参数（config.json → review 段）

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | `deepseek-flash` | DeepSeek V4.1-Flash，支持图片输入；`deepseek-v4-pro` 不支持视觉 |
| `effort` | `low` | 推理力度，low 更快更省，画面判断够用 |
| `frames_per_segment` | `6` | 每段抽帧数（段首事件时刻优先 + 首尾兜底）；窗口按事件密度截取后事件集中在段首，优先覆盖才能数清连杀行数 |
| `max_segments` | `8` | 最多复核段数（从已入选段里取前 N） |
| `max_tokens` | `8000` | 输出上限。推理模型的 reasoning 也占预算，太低会被截断成空回答（v0.3.0 从 3000 调高并加了 reasoning 兜底提取） |
| `concurrency` | `2` | 并发调用数（1-4） |
| `w_signal/w_vlm/w_event` | `0.35/0.45/0.20` | 融合权重：信号分 / VLM精彩分 / 事件权重 |
| `drop_non_highlight` | `true` | VLM 明确判定非高光且置信度 ≥0.7 的段直接丢弃（保底不丢光） |

### 18.4 降级与安全

- **Key 缺失 / 网络失败 / 单段超时**：打印警告后自动跳过复核，按纯信号模式出片，**永远不阻塞成片**（与 Crispy 回退启发式同一哲学）
- 单段失败不影响其他段（每段独立隔离，ThreadPoolExecutor 并发）
- Key 只存在本地 `config.local.json`（gitignored）或环境变量，**不会进 git**；模型只收到关键帧 JPEG，不接收整段视频
- `timeline.json` 的每个片段带完整 `review` 字段（事件类型/精彩分/解说/置信度/token 数），可离线分析

### 18.5 调参速查

| 现象 | 处理 |
|---|---|
| 好片段被 VLM 压分 | `w_vlm` 降到 0.35，或 `drop_non_highlight: false` |
| 死亡回放没被拦住 | prompt 里已写 HUD 知识；可把 `frames_per_segment` 提到 5-6 让模型看到更多上下文 |
| 复核太慢 | `concurrency` 提到 3-4（注意 API 限速），或 `effort` 保持 low、`max_segments` 降到 5 |
| 解说不满意 | 重跑即可（temperature=0.2 有轻微随机性）；或 `frames_per_segment` 加帧让解说更具体 |

### 18.6 三游戏支持（overwatch / csgo2 / valorant）

`--game`（或 GUI 的「游戏」下拉）现在同时驱动**三处**：

| 环节 | overwatch | csgo2 | valorant |
|---|---|---|---|
| L1 Crispy 模型 | `overwatch.npy`（红通道增强） | `csgo2.npy`（边缘+遮罩，需 pillow） | `valorant.npy`（边缘+遮罩，需 pillow） |
| L2 复核提示词 | OW2 HUD 知识（TEAM KILL/死亡回放…） | CS2 HUD 知识（ACE 横幅/残局播报/雷达…） | 瓦罗兰特 HUD 知识（连杀彩条/爆能器…） |
| L2 事件词表 | 团灭/连杀/大招/救援/目标点/单杀 | 五杀ACE/残局/连杀/拆弹/回合攻防/单杀 | 五杀ACE/残局/连杀/大招/拆弹/爆能器/单杀 |
| 启发式 ROI（backend=none） | 右上 0.72,0.05,0.27,0.22 | 右上 0.70,0.06,0.29,0.16 | 右侧中部 0.73,0.24,0.26,0.32 |

```bash
# CS2
python run.py process "xxx.dvr.mp4" --recognizer crispy --game csgo2 --review
# 瓦罗兰特
python run.py process "xxx.dvr.mp4" --recognizer crispy --game valorant --review
```

细节规则：
- **kill_count 升级**：CS2/瓦罗兰特 kill≥5 自动升为「五杀ACE」；OW 沿用 kill≥3 升「连杀」
- **跨游戏词兜底**：模型偶尔输出其它游戏的词（CS2 场景报 team_wipe）会自动映射（team_wipe↔ace、defuse→objective 等）
- **事件权重**：五杀ACE/团灭 1.0 > 残局 0.95 > 连杀 0.9 > 大招 0.8 > 拆弹 0.75 > …（报告「打分公式」段按游戏展示）
- **启发式 ROI 自动切换**：显式传 `--game` 且非 overwatch、且未手动给 `--roi` 时自动套用该游戏默认 ROI；手动 `--roi` 永远优先
- **权重下载**：`python run.py fetch-model --game csgo2,valorant`（一次性）

### 18.7 v0.3.0 召回与截取修复（三杀漏检案例复盘）

真实案例：CS2 1280×960 录像 3:54 的三杀完全没被识别。复盘出**五层叠加缺陷**，全部修复：

| # | 缺陷 | 修复 |
|---|---|---|
| 1 | 4:3 录像被非均匀拉伸到 1920×1080，HUD 奖牌卡片横向变形 33%，MLP 三杀瞬间 p_kill 只有 0.01 | `stretch: "auto"` 按宽高比自动均匀缩放（4:3 → 1440×1080），实测 p_kill 回到 0.98 |
| 2 | crispy 后端完全替代启发式，模型漏检 = 整段漏掉 | **双通道召回**：模型（奖牌框）+ 启发式（kill feed ROI）并行，1.5s 内事件去重合并 |
| 3 | 音频峰值把 209-300s 连成 90s 巨簇，`max_len` 无脑保尾截取，中段三杀被裁掉 | **波次拆分**：按视觉事件间隙（>8s）切开，一波战斗一个候选 |
| 4 | 32s 长波仍只产出一个 16s 窗口，密度最高窗口之外的事件全丢 | **多窗口消耗式切分**：超长波按视觉密度产出至多 3 个窗口，独立打分入选 |
| 5 | VLM 复核抽帧错过 kill feed 三行全在的黄金帧（第三杀上屏时刻） | 关键帧**段首事件优先**采样 + **L1 击杀时间点作为文字提示**喂给 VLM（L1→L2 信息传递）+ `max_tokens` 8000 防推理截断 + reasoning 兜底提取 + 空输出重试 |

修复后同一段录像：三杀段（3:52.38-4:08.38）被正确识别为「连杀×3、VLM精彩 7/10」，解说「连续击杀三人，三杀助力回合胜利」。

### 18.8 OCR 击杀播报读取（v0.4.0）

连杀判定从"VLM 看图猜"变成"OCR 逐帧数行数"。本地离线 RapidOCR（PP-OCRv6）直接读击杀播报文字，拿到**确定性的击杀者/被击杀者/连杀数**：

```bash
pip install -r requirements-ocr.txt          # rapidocr + onnxruntime + opencv，一次性
python run.py process "xxx.mp4" --game csgo2 --recognizer crispy --review
# OCR 默认开启（ocr.enabled=true）；--no-ocr 可关闭
```

工作方式：

| 环节 | 说明 |
|---|---|
| 采样 | `ocr.fps`（默认 1fps）只解 kill feed ROI；`mode=annotate`（默认）只对**入选片段**识别，`mode=full` 全片独立召回 |
| 门控 | 帧差门控跳过播报区无变化的帧；CS2 网络统计条被 ROI 顶部跳过，不再产生假事件 |
| 解析 | OCR 文本框按行聚合，一行 ≥2 个名字 = 击杀条目（左击杀者、右被击杀者，中间武器图标形成宽间距校验） |
| 去重 | 播报在屏停留 ~6s，跨帧重复识别按 (击杀者,被击杀者) 模糊归并（OCR 换字如 DAnTe/DAnTo 不会重复计数）；淡出行被拦腰截断产生的假条目会被同源片段校验剔除 |
| 连杀 | 同一击杀者在 `streak_window`（默认 6s）内连续击杀 = 连杀数，玩家 ID 可配置或从 Crispy 奖牌事件自动推断 |
| 下游 | ① OCR 结果作为**确定性事实**喂给 VLM（连杀数不得低于 OCR 实测）；② VLM 数错时 L3 融合阶段强制修正 kill_count 与事件类型；③ 解说词使用真实玩家名 |

实测（CS2 4:3 录像 3:54 三杀）：OCR 读出 `DAnTe → 英勇无畏的法兰西旗兵 / 暖暖不吃香菜 / 此时若是君在侧` 三行，VLM 据此给出「DAnTe三杀，A点攻势凌厉」；同时一段无击杀的空段被 OCR 负面证据（"未发现任何击杀条目"）正确戳穿，不再产生幻觉解说。

`config.json → ocr` 段参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 依赖缺失时自动跳过并告警，不影响出片 |
| `mode` | `annotate` | `annotate`=只识别入选片段（快，5 分钟录像约 1-3 分钟）；`full`=全片召回（慢） |
| `fps` | `1.0` | 采样帧率，播报停留 ~6s，1fps 足够 |
| `streak_window` | `6.0` | 连杀判定窗口（秒） |
| `player_name` | 空 | 你的游戏 ID；留空则由 Crispy 奖牌事件自动推断 |
| `still_diff` | `2.5` | 帧差门控阈值（低于视为播报区无变化） |

> 安装提示：rapidocr 声明依赖 opencv-python-headless，若与已有 opencv-python 冲突报 `safe-delete` 错误，可 `pip install rapidocr --no-deps` 后单独装其余依赖。

> 日志提示：运行时可能刷出大量 `[RapidOCR] The text detection result is empty` WARNING，**属正常现象可忽略**——OCR 通道逐帧读 kill feed ROI，多数采样时刻播报区是空的（无击杀发生），检测模型返回 0 个文本框时 RapidOCR 就会打这条日志。代码已对空结果做防护（`_parse_rows` 对空 `txts` 返回空列表），流程不受影响。若嫌刷屏，可在 `owagent/ocr.py` 的 `detect_ocr` 里把 `logging.getLogger("RapidOCR").setLevel(logging.WARNING)` 改为 `logging.ERROR`。

> 日志提示 2：启动 OCR 相关通道时还会刷出多串 `[INFO] [RapidOCR] Using ...onnx` / `File exists and is valid`，夹杂 `[32m` `[0m` 之类的乱码——**同样是正常 INFO 日志，不是报错**：前者是加载本地 onnx 模型（ocr / skull / killcard 三个通道各自建引擎，会重复出现 2-3 遍），后者是确认模型文件已离线就绪；乱码是 ANSI 颜色码经过管道输出未渲染，纯属视觉噪音。（v0.7.1 已修复：RapidOCR 初始化会重置 logger 级别，静音必须在引擎创建之后设置，现三处均已生效，INFO 不再刷屏。）

### 18.9 OW 屏幕中心击杀骷髅头识别（v0.5.0）

OW2 在**击杀确认时于准星附近显示一个红色小骷髅头**（约 60-90px，持续 0.3 秒后淡出）。关键特性：**该提示只对击杀者本人显示**——击杀播报里全是全场的击杀，而这个骷髅头只属于录像玩家，是全游戏最专属的"我的击杀"信号（助攻淘汰也会显示）。

新增第四召回通道 `owagent/skull.py`（仅 overwatch 生效，默认开启）：

| 环节 | 说明 |
|---|---|
| 采样 | 16fps 只解码准星区裁剪（340×300 @1080p，按分辨率缩放），速度 ~85s/4min 视频 |
| 签名 | HSV 纯红掩码 → 连通域 → **眼窝双孔签名**：红色实心块内两个 8-40px 的孔（左右眼窝），dx 20-65px、dy<16px、大小相近。爆炸/敌方红描边/血条的随机孔洞极难同时满足该几何约束 |
| 过滤 | 生存期 >1.3s 的块是静态物（发光机械眼等），剔除；击杀瞬间骷髅与红色命中标记 X 合并成大块，尺寸上限放宽到 150px |
| 确认 | 候选时刻读击杀播报（t-1.5 与 t+1.2/+2.5 两次，防 OCR 偶发漏读）：出现**新的播报行** → 升级为确认玩家击杀（strength 0.95，附带击杀者/被击杀者名字）；无新行 = 可能是助攻淘汰或噪声，保留为弱事件 |
| 打分 | timeline 打分新增**玩家击杀加成**：含确认击杀的候选段 +0.10，每多一杀 +0.05（封顶 +0.20）——"我的击杀"比通用播报变化更值得进集锦 |

实测（OW 录像 3:45，玩家 LDWMID 全场 5 杀含两次双杀）：

- 骷髅头通道检出全部 5 个击杀时刻中的 5 个（169.0/181.8/183.9/206.2/207.8s），其中 4 个经击杀播报确认并附带击杀者/被击杀者名字
- 此前击杀段因无音频峰值（cross=0）打分偏低落选；加入击杀加成后，**双杀段（02:48-03:04）与 3:25 击杀段（03:24-03:40）双双入选**，VLM 复核给出「末日铁拳开大进场，成功击杀一人」「裂地猛击命中，补杀一人」（精彩度均 7/10）

`config.json → skull` 段参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 仅 overwatch 生效；opencv 缺失时自动跳过并告警 |
| `fps` | `16` | 采样帧率；骷髅头可见约 0.3s，16fps 能保证至少 2-4 帧命中 |
| `confirm_feed` | `true` | 用击杀播报 OCR 确认玩家击杀并附带名字（需 rapidocr） |

### 18.10 CS2 底部击杀卡牌识别（v0.6.0）

CS2 在**底部中央、段位徽章右上方**显示本回合击杀卡牌堆——每杀一张卡，扇形向右展开，**最新卡（最右）的数字 = 本回合击杀数**，整回合持续显示。和 OW 骷髅头同理：击杀播报是全场的，卡牌堆**只计录像玩家自己的击杀**，且数字本身就是回合连杀数，无需推断。

新增第五召回通道 `owagent/killcard.py`（仅 csgo2 生效，默认开启）。核心设计——**不连续扫描，只在其他召回通道的事件时刻前后读卡牌数字做差分**（任何击杀都会产生播报行/奖牌事件，门控必然触发；卡牌数字只随玩家自己的击杀增长）：

| 环节 | 说明 |
|---|---|
| 阶梯读数 | 每个门控组在 -0.7/-0.3/+0.5/+1.2/+2.2s 五个 offset 各读一次。负 offset 抓「弹出窗口早于门控」的击杀（启发式墨迹检测滞后 ~0.5s），长正 offset 抓慢热卡（部分击杀的卡 +2s 才可读） |
| 最右卡取值 | 卡牌向右扇形展开（卡 k 在卡 k-1 右侧 ~17px），每帧取**最右侧有效数字框** = 当前回合计数；置信度 <0.85 或数字 >5（5v5 回合上限）直接丢弃，天然过滤 "7"/"WU" 类误读 |
| 状态机 | 跨门控单调递增判定：数字没变 = 别人的击杀；数字变大 = 玩家击杀（增量即杀数）；数字变小 = 回合重置。**回合结束判定只用"空读证据"**（连续 ≥2 组读不到卡牌且距上张卡 >8s），不用门控间隔——门控稀疏 ≠ 卡牌消失 |
| 阶梯内递增 | 单个门控窗口内看到 1→2→3 的递增序列 = 这些卡就在窗口内弹出（如 2.5s 内的三连杀），kills = 终值-首值+1 |
| 下游 | 事件带 `confirmed=True` + `kills` 增量 + `streak` 回合累计数：timeline 享受玩家击杀加成（每杀 +0.05 封顶 +0.20）；reviewer 把「玩家在此段落地 N 杀」作为 kill_count 硬下限喂给 VLM 并在 L3 强制修正 |

实测（CS2 4:3 录像，玩家两回合共 5 杀含 3:52-3:54 战斗的回合内 3 杀）：

- 卡牌数字全程被正确跟踪：174.4s（第 1 杀）、187.8s（第 2 杀，弹出窗口早于门控 0.5s，靠负 offset 阶梯抓到）、231.9s（回合重置后 2 杀 + streak=3）
- 241s/258s 等数字未变的门控全部正确跳过（无假阳性）；回合 A→B 之间的 14 组空读正确触发回合重置
- 3:52-3:54 战斗段获得击杀加成 +0.15 并稳定入选

`config.json → killcard` 段参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 仅 csgo2 生效；rapidocr 缺失时自动跳过并告警 |
| `roi` | 见 config | 卡牌区比例坐标（1280×960 4:3 实测；16:9 未实测，偏移时用 roi-preview 微调） |

> 性能提示：阶梯读数约 5 次 OCR/门控组（本例 48 组 = 240 次 ≈ 2 分钟，叠加在 OCR annotate 之上）。CS2 长录像若嫌慢，可把 `killcard.enabled` 关掉——五通道里它是精度增强项，召回由其他通道兜底。

### 18.11 成片画质修复（v0.7.0）

旧默认渲染参数有四层叠加画质损失，成片明显糊于源录像：

1. **分辨率降级**：默认 1280×720——1080p 源只剩 56% 像素；1280×960 的 CS2 4:3 源被缩到 960×720 还加黑边
2. **帧率减半**：默认 fps=30，60fps 录像被强制降帧，快节奏甩镜头顿挫
3. **压缩档位低**：crf 23 + veryfast 是 x264 默认档，运动剧烈的枪战画面块效应可见
4. **双重有损编码**：烧字幕时切片先编码一代、烧录再编码一代，两代损失

修复内容：

| 项 | 旧默认 | 新默认 |
|---|---|---|
| `width`/`height` | 1280/720 | **0/0 = 跟随源**（无缩放无 letterbox，与源同分辨率时跳过 scale/pad 滤镜） |
| `fps` | 30 | **0 = 跟随源**（60fps 保持 60fps） |
| `crf` | 23 | 18（接近视觉无损） |
| `preset` | veryfast | medium |
| 烧字幕编码链 | 切片 crf23 → 烧录 crf23（两代有损） | 切片 crf10 近无损中间档 → 烧录 crf18（**仅一代可见有损**） |
| 音频码率 | 128k | 192k |

实测：OW 1080p60 成片 1920×1080@60fps、码率约 10 Mbps；CS2 4:3 源成片 1280×960 原样保持、无黑边；与源逐帧并排目检仅轻微差异，HUD 文字锐度一致；烧字幕失败时自动把近无损中间件回落转码到最终档（不再直接交付超大文件）。

兼容性：显式给 `width/height/fps` 正数仍按指定值缩放/降帧，老配置行为不变。`render` 节新增校验（crf 0~51、preset 合法性、尺寸非负）。

### 18.12 P1 识别精度评估与窗口兜底修复（v0.7.2）

用两段已知真值的测试录像（CS2 5min / OW 3:45）开 VLM 复核做全链路评估：

| 层 | 指标 | CS2 | OW |
|---|---|---|---|
| L1 玩家击杀召回 | killcard/骷髅头确认击杀 → 候选段覆盖 | 4/4（修复后） | 4/4 |
| L1 片段精度（复核前） | 入选段中真实高光占比 | 3/6 = 50% | 2/5 = 40% |
| L2 VLM 判定 | 事件类型/杀数/精彩度 | 三杀 kills=3 exc=7 ✓ | 大招 kills=4 exc=8 ✓ |
| L2 假阳性拦截 | 非高光段被正确丢弃 | 3 段（含 2 段纯架枪） | 3 段（含 1 段非对局画面） |
| L3 最终精度 | 入选段含玩家击杀 | 2/2 = 100% | 2/2 = 100% |

**修复的缺陷**：超长波贪心切分达到 5 窗口上限后，剩余未覆盖事件被**静默丢弃**——实测 76s 大波 25 个视觉事件中第 6、7 个（187.75/189.4s 玩家击杀）落得没有候选段覆盖，直接导致击杀从成片消失。修复：窗口循环结束后为每个漏网事件补一个专属窗口（`timeline.py` 兜底逻辑），保证**任何视觉事件至少落在一个候选段里**。

**发现的待改进点**（后续路线输入）：
1. **玩家归属跨通道矛盾**：killcard（本地玩家专属信号）确认的击杀，OCR/VLM 却归到别人名下（如 DAnTe 的杀被读成 "JILaOSh1s"/"Jaden"）——需要玩家名规范化与互证层
2. **重叠合并截断风险**：相邻段粘合并 `start = end - max_len` 截断时可能削掉前段的事件（实测一次仅剩 0.1s 余量）；应按事件时刻锚定截断点
3. **VLM 关键帧覆盖**：复核帧均匀抽样，击杀瞬间帧可能没被抽中，导致精彩度低估

### 18.13 玩家中心性加权与玩家名互证（v0.8.0）

解决 18.12 发现的头号短板"复核前片段精度 50%"：一半预选段是别人的高光或纯架枪画面，全靠 VLM 复核兜底。

**A · 玩家中心性加权**（`timeline.py` + `select.playerless_penalty`）：
- 击杀卡牌/骷髅头是 HUD 私有元素（只有录像主人能看到），一旦全片确认过"玩家本人击杀"，**无确认的候选段统一降权 0.15**（可配，0=关闭）
- 玩家击杀段原有 +0.10~0.20 加成不变——排序位差拉大到 0.25~0.35，别人的双杀再热闹也挤不进前排
- 整片没确认过玩家击杀时（通道关闭/玩家零杀）**不降权**，纯音频精彩段不受误伤

**B · 玩家名互证**（`ocr.py` / `reviewer.py`）：
- `_infer_player` 推断锚点扩展：Crispy 奖牌事件之外，新增 killcard/骷髅头确认时刻（±2.5s 内的播报行给该击杀者投票）——**玩家改名也不怕**（实测 CS2 录像玩家回合 A 叫 j1LLa0Sh1、回合 B 改名 DAnTe，锚点投票仍锁定 DAnTe）
- 玩家簇内拼写统一：`DAnTo/DAnTe` 类 OCR 换字归一到最高频写法；配置手输名模糊映射到 OCR 簇（大小写/个别字符差异自动对齐）
- VLM 复核提示词注入玩家名：「录像玩家是『DAnTe』，kill feed 中该名字的行就是玩家本人的击杀」——杜绝解说词张冠李戴（此前 DAnTe 被叫成 "Jaden"）
- kill_count 下限新增 `confirmed_kills` 维度（骷髅头确认的击杀也计入）；重叠段合并后按最终窗口**重算**确认击杀数（修复合并截断把窗口外卡牌击杀也计入的虚增 bug）

**实测效果**（同两段测试录像，v0.7.2 → v0.8.0）：

| 指标 | CS2 | OW |
|---|---|---|
| 预选段玩家击杀覆盖 | 4/4 | 5/5（v2 为 4/5） |
| 玩家击杀段 vs 无杀段分差 | 0.25~0.35（原 0.10~0.15） | 同左 |
| VLM 解说词玩家名 | 「DAnTe连杀三人，A区突破犀利」✓ | 「LDWMD末日铁拳双杀」✓（v2 曾误称 Jaden） |
| 最终成片 | 2 段纯玩家高光（32s） | 3 段（39.8s），杀数计数 2/3 全对（v2 漏计 2 杀） |

### 18.14 红框确认与观战视角剔除（v0.9.0）

18.13 之后遗留一个根本性误判：**"击杀卡牌/红框 = 录像玩家的击杀"只在本人视角成立**。CS2 死亡观战（或 GOTV 回放）时，播报红框与击杀卡牌 HUD 跟随的是**被观看的队友**——他们击杀时卡牌照样 +1、播报行照样红框，旧逻辑会把队友的连杀记成你的高光（实测 5 分钟 CS2 录像里，176.5s/187.75s 两组"确认击杀"其实都是被观战队友 J1La0Sh1/阿晴ingggggg 的击杀）。

**新击杀归属标准**（用户定义）：击杀属于目标用户 ⟺ 右上击杀播报行有**红框** ∧ **卡牌/骷髅头变化** ∧ **本人视角**——不看播报行左边的击杀者名字，也不推断"接下来是不是被击杀画面"。

**A · 击杀红框通道**（`owagent/redbox.py`，仅 csgo2）：
- 当前视角玩家的击杀行会整行红色高亮：击杀后 0-1s 是实心红填充（可滞后至 ~2.5s），随后稳定为细红描边持续 5-10s
- 与 killcard 同一批门控时刻做阶梯读数（`-0.6s ~ +3.0s` 共 9 点），纯像素无 OCR（每帧 ~50ms）
- 判定特征：**右缘锚定的近满宽红色横条**（水平 21px 连续红腐蚀 + 右缘 30px 条带内 ≥20 红像素的扫描行 ≥4）——砖墙红纹理、烟雾、白闪全部天然免疫（实测 37 帧标注序列 37/37 完美分离）
- **红框×卡牌互证**：卡牌有差分但无红框 → 撤销确认（误读防护）；有红框但卡牌无差分 → 只作召回（滞留红框不虚增击杀）；红框通道全程零命中视为通道失效，不降级

**B · 视角判定通道**（`owagent/pov.py`，仅 csgo2）：
- **观战控制条**：底部「[MOUSE1] 下一名玩家 [CTRL] 总览」提示条只在观战时出现——全片 2fps 扫描底部条带的亮文字密度（纯 numpy，~20s/5min），购买菜单残影与观战条通过文字行数/亮度双阈值区分
- **底部信息卡名牌**：观战时信息卡显示被观战者名字+武器（本人第一人称不显示名字）——观战区间内每 4s + 每个击杀锚点时刻 OCR 名牌，得到"当前被观看者"时间线（左端名字=被观战者，过滤武器/皮肤/拾取词）
- **目标用户推断**：`ocr.player_name` 优先；缺省用**本人视角锚点**（无观战条的击杀确认时刻）的播报击杀者名投票——观战锚点不参与投票，天然排除被观战队友的名字
- **两重剔除**：① 观战他人视角的击杀确认被撤销（事件带 `spectate` 标注，`demote_spectated`）；② 观战区间整段从候选段切掉（`build_timeline` 新增 `exclude_intervals`，按区间减法切窗口、过滤区间内事件）
- 观战目标用户本人的区间（GOTV 回放自己的 demo）不剔除——名牌与目标用户名模糊匹配（阈值 0.6，兼容 OCR 换字）

**C · 帧抓取偶数尺寸修复**：ffmpeg rawvideo 会把奇数宽高的裁剪输出补齐到偶数，长度校验失败导致整条通道静默零命中——所有通道 ROI 宽高统一过 `even()`（本次 redbox/pov 两个新通道都踩中，靠"通道零命中时定位"发现）。

`config.json → redbox / pov` 段参数：

| 键 | 默认 | 说明 |
|---|---|---|
| `redbox.enabled` | true | 击杀红框通道开关（仅 csgo2） |
| `pov.enabled` | true | 视角判定通道开关（仅 csgo2） |

> 提示：红框/视角通道 ROI 均按 1280x960 4:3 实测标定，16:9 分辨率未验证（控制条/名牌相对位置应该一致，偏移可用 `pov.bar` / `pov.card` 覆盖）。OW 走骷髅头通道不受影响。

**实测效果**（同 CS2 测试录像，v0.8.0 → v0.9.0）：

| 指标 | v0.8.0 | v0.9.0 |
|---|---|---|
| 目标用户击杀确认 | 4 组（含 2 组误判：观战队友的击杀被记成玩家的） | 仅 DAnTe 本人的 2 组（231.9/233.7） |
| 观战视角处理 | 无（观战段照常入选） | 151.5-195.0s 观战段整段剔除，176.5/187.75 确认撤销（标注被观战者 J1La0Sh1/阿晴ingggggg） |
| 目标用户推断 | 锚点投票（观战击杀污染票池） | 仅本人视角锚点投票 → DAnTe，零污染 |
| 成片 | 含队友视角"你的高光" | 5 段全部本人视角；最高分段 = DAnTe 三杀（卡牌确认、本回合累计 3 杀） |

### 18.15 OW 名牌通道：目标用户直读 + 名字×播报协同（v0.9.1）

CS2 有了红框×卡牌×视角三重确认后，OW 路径的目标用户识别仍靠**播报投票推断**——骷髅头/奖牌时刻附近的击杀者名投票，票池可能被误读污染（LDWMID 曾被推断成 LDWMD 之外的写法，甚至误报 Jaden）。用户指定新方案：**左下角名牌直接读用户名，与右上角击杀播报协同判断**。

**A · 名牌通道**（`owagent/nameplate.py`，仅 overwatch）：
- OW2 左下角名牌（英雄头像+等级+名字）是**唯一持续印着录像者名字的 HUD**——0.5fps 帧差扫描（静态帧跳过 OCR），读数剥离粘连等级数字（"1LDWMD"→"LDWMD"）、过滤血量对/界面词/战斗噪声后模糊聚类（阈值 0.72，LDDWMD/LOWMD/TLDWMD 归一簇），最高频拼写即目标用户名；最少 5 次读数才采信（防单帧噪声）
- 读到的名字注入 `ocr.player_name`（用户已配置时不覆盖）——播报 own 判定、玩家名归一、VLM 玩家提示、`tl["player"]` 全链路自动受益
- **名牌同时是视角信号**（同 CS2 底部信息卡）：观战他人时换成被观战者名——连续 ≥3 次同一非主名读数 → 他人视角区间整段剔除（实测本片无此区间，机制保留）

**B · 名字×播报协同**（`cross_check_killer` / `confirm_feed_rows`）：
- 骷髅头确认事件自带 killer 名（读播报行所得）：与名牌名匹配（≥0.45）→ 双信号互证保持确认；不匹配 → 撤销确认（形状误检/播报误读防护）
- full OCR 模式下：播报行击杀者名=名牌名 → 直接升级为确认玩家击杀（播报印着所有人的击杀，名字对上就是本人的）

**实测效果**（OW 测试录像，v0.8.0 → v0.9.1）：

| 指标 | v0.8.0 | v0.9.1 |
|---|---|---|
| 目标用户名 | 播报投票推断（LDWMD，票池有污染风险） | 名牌直读 LDWMD（85-225s 共 100+ 次读数，零投票） |
| 名字来源 | 间接（骷髅头时刻的 killer 名） | 直接（名牌常驻 HUD） |
| 击杀确认 | 仅骷髅头单信号 | 骷髅头 + 名字匹配双信号互证 |

`config.json → nameplate` 段：`enabled`（默认 true）、`roi`（1920x1080 实测，可覆盖）、`fps`（默认 0.5）。
