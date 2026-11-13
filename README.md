# OW Highlight Agent · P0 本地原型

自动识别《守望先锋》录像里的**击杀播报/高光信号**，把候选片段打分、排序，再用 FFmpeg 剪成一段集锦，并输出时间轴、字幕和决策报告。

> **两条识别路线**：默认走「HUD 像素信号 + 音频能量」的零依赖启发式；可选启用 **Crispy 预训练神经网络**（第 16 节）——真实游戏画面训练的击杀检测模型，纯 numpy 推理，支持守望先锋 / CS2 / 瓦罗兰特。启发式路线**不认识英雄、不读文字、不理解击杀语义**，目的是用最小依赖把「采集 → 召回 → 打分 → 时间轴 → 剪辑 → 反馈」这条链路完整跑通。

**本机实测结果**（Windows 11 / Python 3.13.14 / FFmpeg 9.0.2）：

| 项目 | 结果 |
|---|---|
| 合成测试录像 | 30s，8 次模拟击杀播报 |
| 视觉事件召回 | 8 / 8，召回率 **100%**，精确率 **100%**（±1.0s 容差） |
| 音频峰值召回 | 8 / 8 |
| 成片 | 3 段无重叠片段、25.26s、H.264 1280×720@30fps + AAC |
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
| `stretch` | false | 4:3 拉伸画面（如 CS 玩家的 1440×1080）设为 true，HUD 框坐标会随之调整 |

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
| `max_total_sec` | 90.0 | 成片总时长上限 |
| `min_score` | 0.0 | 得分下限，用来砍掉低质量候选 |

### `render` — 输出

| 字段 | 默认值 | 说明 |
|---|---|---|
| `width` / `height` | 1280 / 720 | 横版输出分辨率（保持原比例 letterbox） |
| `fps` | 30 | 输出帧率 |
| `crf` | 23 | H.264 质量，越小越清晰越大 |
| `preset` | veryfast | 编码速度档 |
| `vertical` | false | true 输出 9:16 竖版 1080×1920（居中裁切） |
| `burn_subtitles` | false | 是否把 SRT 烧进画面（需 libass，失败会自动跳过） |
| `bgm` | null | BGM 文件路径 |
| `bgm_volume` | 0.18 | BGM 混音音量 |

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

流程：ffmpeg 先把任意分辨率的帧**归一到 1920×1080**（模型训练时的基准；4:3 拉伸画面用 `stretch: true`），裁出对应游戏的 HUD 框 → 逐帧预处理 → MLP 输出 `p_kill` → **`p_kill` 越过 `confidence` 的上升沿记为一次击杀事件**（`source: "model"`）→ 进入和启发式完全相同的聚类、打分、剪辑流水线。`signals.csv` 在此模式下记录每帧 `t / p_kill`，界面曲线图会画出概率与阈值线。

### 16.3 参数与调参

| 现象 | 处理 |
|---|---|
| 整场 `p_kill` 都贴着 0 | 画面不是对应游戏的 16:9 全屏画面；确认录像无黑边、无窗口边框、游戏模型选对 |
| 击杀漏检 | `--confidence` 降到 0.4~0.5 |
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
