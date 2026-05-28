# Video_Interpretation

把电影预告片自动合成为 B 站影评/解说视频的 CLI 工具。

## 是什么

输入一个原版预告片 + 一段中文解说稿，输出一个带配音、背景音乐和双字幕（原英文 + 中文解说）的成品视频，可直接发布到 B 站。

```
原视频 (webm/mp4) ──┐
中文解说 (人工/LLM)  ├──> CLI 流水线 ──> 成品 mp4（含双字幕、配音、BGM）
背景音乐 (mp3)      ──┘
```

## 快速开始

### 环境要求
- Python 3.11+
- ffmpeg（在 PATH 中）
- Windows（已验证 git-bash）/ macOS / Linux
- 可选：LLM API key（仅使用 `llm-narrate` 时需要）。默认 provider 是 `openai_compatible`（智谱 GLM），认 `ZHIPU_API_KEY`；切到 `anthropic` 则认 `ANTHROPIC_API_KEY`。具体 env var 名由 `config.toml` 的 `llm.api_key_env` 控制。

### 安装

```bash
pip install -r requirements.txt
```

### 5 分钟体验

```bash
# 1. 把一个预告片加入项目
python main.py init my_first --video "D:/dl/some_trailer.mp4" --title "我的第一部"

# 2. 识别英文字幕
python main.py transcribe my_first

# 3. 写中文解说稿（编辑器打开 projects/my_first/narration.txt）
# 4. 标时间轴（编辑器打开 projects/my_first/narration_aligned.txt，参照 subtitle.srt 时间）

# 5. 合成配音
python main.py tts my_first

# 6. 合成成品视频
python main.py build my_first

# 成品：projects/my_first/output.mp4
```

## 目录结构

```
Video_Interpretation/
├── main.py / config.toml / requirements.txt
├── vi/                          # 源代码包
│   ├── cli.py / config.py / paths.py
│   ├── commands/                # 各子命令
│   └── core/                    # 算法核心（whisper、TTS、ffmpeg、LLM）
├── projects/                    # 每个视频一个子目录
│   └── <name>/
│       ├── project.toml         # 项目级配置
│       ├── source.<ext>         # 原视频（init 时复制/链接进来）
│       ├── subtitle.srt         # 英文字幕（transcribe 产出）
│       ├── narration.txt        # 中文解说（人工写 / LLM 生成）
│       ├── narration_aligned.txt# 带时间的解说（人工对时 / LLM 草稿）
│       ├── narration_subtitle.srt # 中文字幕（tts 产出）
│       ├── visual_description.txt # 关键帧画面描述（vision 模型产出，可选）
│       ├── voice.wav            # 配音（tts 产出）
│       └── output.mp4           # 成品
└── shared/                      # 跨项目资产
    ├── bgm/default.mp3
    └── whisper/                 # whisper-faster.exe + 模型
```

## 命令参考

所有命令格式：`python main.py <command> [project_name] [options]`

项目名可省略 —— 当 cwd 在某个 `projects/<name>/` 目录里时，自动定位。

### init — 创建项目

```bash
python main.py init <name> --video PATH [--title T] [--link]
```

- `name` ASCII 字母数字加 `_-`
- `--video` 源视频绝对路径
- `--title` 中文显示名（可选）
- `--link` 用软链而非复制（节省磁盘）

### ls — 列出项目

```bash
python main.py ls [-l]
# -l 显示每个项目的完成阶段：[transcribe+ refine+ narration+ aligned+ tts+ build+]
```

### transcribe — 识别英文字幕

```bash
python main.py transcribe [name] [--engine py|exe] [--model tiny|base|small|...] [-f]
```

`--engine py` 用 faster-whisper 库（默认）；`--engine exe` 用 whisper-faster.exe（速度更快，但只在 Windows shared/whisper/ 下可用）。

### refine — 重新切分字幕

```bash
python main.py refine [name] [-f]
```

从 `source.json` 用更精细的算法重新切分 subtitle.srt，避免一行字幕过长。

### llm-narrate — LLM 生成解说稿

```bash
python main.py llm-narrate [name] [--style S] [--length N] [--align] [--srt]
                           [--model M] [--provider anthropic|openai_compatible]
                           [--base-url URL]
                           [--vision] [--vision-model M] [--frame-interval SEC]
                           [--force-vision] [-f]
```

- `--style` 风格关键词，覆盖全局默认（例："恐怖片专题，强调氛围"）
- `--length` 目标字数（默认 800，仅两步流程使用）
- `--align` 跑完 `narration.txt` 后顺带产出 `narration_aligned.txt` 草稿
- `--srt` 一步式：直接产出 `narration_aligned.txt`，跳过中间 `narration.txt`
- `--provider` LLM provider，覆盖配置（`openai_compatible` 默认 / `anthropic`）
- `--base-url` OpenAI 兼容接口的 base URL（如自建 Ollama / SiliconFlow / 智谱）
- `--vision` 抽取关键帧 + 调多模态视觉模型描述画面，把描述拼进 prompt
- `--vision-model` 视觉模型名（默认从配置读，回退 `glm-4v-flash`）
- `--frame-interval` 抽帧间隔（秒，默认 3）
- `--force-vision` 即使 `visual_description.txt` 已存在也重新分析

**任何 `--align` / `--srt` LLM 草稿都必须人工 review 后才能跑 tts。**

API key 来自 `[llm].api_key`（config.toml）或 `[llm].api_key_env` 指向的环境变量。

### align — 对齐时间轴

```bash
python main.py align [name] [--llm] [-f]
```

不带 `--llm` 时生成空模板让你手填；带 `--llm` 时由配置的 LLM 出草稿。

### preview-srt — 不跑 TTS 预览字幕节奏

```bash
python main.py preview-srt [name]
```

按字符数均分把 `narration_aligned.txt` 转成预览 SRT，用来检查时间分配是否合理。

### tts — 合成配音 + 中文字幕

```bash
python main.py tts [name] [--voice V] [--volume +0%] [-f]
```

用 edge-tts（默认音色 `zh-CN-YunjianNeural`）。如果某段配音超过分配时间窗口，会自动提高语速重合成。

### build — 混音 + 烧字幕

```bash
python main.py build [name] [--bgm PATH] [-f]
```

用 moviepy 混合三轨（配音 1.4 倍 + 原音 0.15 倍 + BGM 0.15 倍），再用 ffmpeg 烧入英文（小号置顶）+ 中文（大号置底）双字幕。产出 `output.mp4`。

### all — 一键串跑

```bash
python main.py all <name> [--from STEP] [--skip a,b] [-f]
```

依次跑 transcribe → refine → tts → build。如果发现没有 `narration_aligned.txt` 会停下提示你先写。

### clean — 清理中间产物

```bash
python main.py clean [name] [--all] [--dry-run]
```

默认删除 `_tts_tmp/`、`voice.wav`、`narration_subtitle.srt` 等中间产物。`--all` 连 `subtitle.srt` / `source.json` / `output.mp4` 也删（**保留** `narration*` 文件，因为是手写的）。

## 配置

### 全局 `config.toml`

放在仓库根目录。涵盖 ASR/TTS/混音/字幕样式/编码等所有默认值。常用字段：

```toml
[tts]
voice = "zh-CN-YunjianNeural"   # 试 zh-CN-XiaoxiaoNeural（女声）/ zh-CN-YunyangNeural 等
volume = "+0%"
max_rate_boost = 40              # 配音超时时最高加速 40%

[mix]
voice_gain = 1.4
ambient_gain = 0.15              # 原视频环境声
bgm_gain = 0.15
bgm_path = "shared/bgm/default.mp3"

[subtitle_narration]
font = "Microsoft YaHei"
size = 26
primary = "&H00FFFF"             # ASS 颜色：黄色
alignment = 2                    # 2=底部居中，8=顶部居中

[fonts]
# Linux 上 Windows 字体不存在时的 fallback，代码自动启用
linux_subtitle_original = "DejaVu Sans"
linux_subtitle_narration = "Noto Sans CJK SC Regular"

[llm]
provider = "openai_compatible"   # 或 "anthropic"
base_url = "https://open.bigmodel.cn/api/paas/v4"   # 智谱开放平台
model = "glm-4.7-flash"
api_key = ""                     # 直填（不推荐进 git）；或留空走 api_key_env
api_key_env = "ZHIPU_API_KEY"    # provider=anthropic 时改 "ANTHROPIC_API_KEY"
vision_model = "glm-4.6v-flash"  # --vision 时使用
default_style = "B站电影解说，先抑后扬，三段式：背景→亮点→收尾"
default_length = 800
n_segments = 3                   # --align 时切几个时间块
```

### 项目 `project.toml`

每个项目一份，**只填覆盖全局的字段**。最小：

```toml
[project]
name = "hungry_2026"
title = "饥饿《Hungry》2026 预告片"

[source]
video = "source.webm"

[output]
filename = "output.mp4"
```

如要覆盖：

```toml
[tts]
voice = "zh-CN-XiaoxiaoNeural"   # 这个项目用女声

[mix]
bgm_path = "shared/bgm/horror.mp3"  # 用专门的恐怖片 BGM
```

**优先级**：命令行参数 > project.toml > config.toml。

## 解说稿写作指南

### 人工写稿

`narration.txt` 是纯文本，每段一个意群即可：

```
今天给大家带来一部 2026 年最新的怪兽惊悚片《Hungry》。
故事发生在美国南部的幽暗沼泽，一群年轻人误入巨型河马统治的禁区...
```

`narration_aligned.txt` 是 SRT 风格，每个 block 一段：

```
1
00:00:02,000 --> 00:00:33,000
今天带来一部 2026 年最新怪兽惊悚片《Hungry》。河马，每年造成超过五百人死亡...

2
00:00:35,000 --> 00:01:11,000
当游船驶入沼泽深处，水面突然爆开...
```

**对时建议**：
- 在编辑器里同时打开 `subtitle.srt`（英文字幕的时间轴），参照剧情转折切分
- 一段建议覆盖 10-40 秒，太短配音容易破碎，太长容易跟不上画面
- 每段字数控制在 60-120 字（视语速决定）
- 段与段之间留 1-3 秒空隙，避免配音叠在画面停顿上

### LLM 生成

```bash
# 两步流程：先 narration.txt，再 --align 对时
python main.py llm-narrate my_proj --style "悬疑恐怖向，强调氛围" --length 700 --align

# 一步流程：直接产出 narration_aligned.txt（SRT 草稿）
python main.py llm-narrate my_proj --srt --style "悬疑恐怖向"

# 加视觉理解：先抽帧 + 调多模态模型描画面，再喂给文本模型
python main.py llm-narrate my_proj --srt --vision --frame-interval 3
```

- LLM 出的 `narration.txt` 通常可用，可能需要小幅微调语感
- LLM 出的 `narration_aligned.txt` 是**草稿**，必须人工 review：
  - 时间分配是否匹配画面节奏
  - 是否有剧透、人名错误
  - 段落字数是否过长（会被 tts 自动加速到失真）
- `--vision` 第一次会调用视觉模型抽帧、产出 `visual_description.txt` 缓存；后续重跑会复用缓存，除非加 `--force-vision`

## TTS 调优

### 换声音

[edge-tts 中文音色列表](https://github.com/rany2/edge-tts)，常用：

| 音色 | 特点 |
|---|---|
| `zh-CN-YunjianNeural` | 男声，沉稳（默认） |
| `zh-CN-YunyangNeural` | 男声，新闻播报感 |
| `zh-CN-XiaoxiaoNeural` | 女声，自然 |
| `zh-CN-XiaoyiNeural` | 女声，活泼 |

在 `project.toml` 改 `[tts] voice` 即可。

### 超时自动加速

如果某段解说稿字数超过分配的时间窗口，TTS 引擎会先按正常语速合成，发现超时后**自动重新以更高语速合成**（上限 40%，可在 `config.toml` 调 `max_rate_boost`）。如果即使到上限仍超时，会被截断 —— 此时应缩短解说稿或加长时间窗口。

## 字幕样式调优

ASS 样式参数速查（在 `config.toml` 的 `[subtitle_original]` / `[subtitle_narration]` 修改）：

| 字段 | 说明 |
|---|---|
| `font` | 字体名（系统已安装的） |
| `size` | 字号 |
| `primary` | 字体颜色，ASS 格式 `&HBBGGRR`（注意 BGR 顺序，不是 RGB） |
| `outline` | 描边粗细 |
| `margin_v` | 距上/下边距像素 |
| `alignment` | 1-9，数字小键盘布局（2=底中，5=正中，8=顶中） |

## 多项目管理

### 项目命名

- 用 ASCII 字母数字加 `_-`，避开中文路径
- 推荐 `<片名缩写>_<年份>`，如 `hungry_2026` / `scream_2026`
- 中文显示名放 `project.toml` 的 `title`，只用于 ls 列表显示

### 共享资产

- `shared/bgm/` 放多个 BGM，per-project 通过 `[mix].bgm_path` 选用
- `shared/whisper/` 放 whisper-faster.exe（如使用 exe 引擎）
- `.cache/hub` 放 HuggingFace 模型缓存（faster-whisper Python 引擎下载到此）

## 常见问题

**Q: ffmpeg 报 "No such filter" 或 SRT 路径解析错误**
A: Windows 盘符冒号会被 ffmpeg subtitles 滤镜误解。`build` 命令已通过把 SRT 临时拷贝到无冒号相对路径绕开，无需手动处理。如果你自定义流程，注意保留这个 workaround。

**Q: faster-whisper 模型下载慢**
A: 设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

**Q: edge-tts 偶发 403 错误**
A: 网络抖动或微软接口临时限流，重试即可（脚本会从头重跑该段）。

**Q: moviepy 写视频失败**
A: 通常是 ffmpeg 版本太老。升级到 ffmpeg 6.0+。

**Q: LLM 生成的解说稿字数偏少 / 偏多**
A: 调 `--length` 参数；也可以在 `--style` 里强调"语速要快/慢"。

**Q: 1M context 这种参数我用不上吗**
A: 本工具的 LLM 调用只读字幕（几 KB）+ 解说稿，不会触及大 context；走 `anthropic` provider 时 prompt caching 启用后多次重跑（如调风格）成本低。

## 从老版本迁移

如果你保留了根目录的旧脚本（`tts.py` / `make.py` / `transcribe.py` 等），它们已被这套 CLI 取代。映射关系：

| 旧脚本 | 对应命令 |
|---|---|
| `transcribe.py` | `python main.py transcribe <name>` |
| `refine_subtitle.py` | `python main.py refine <name>` |
| `build_narration_srt.py` | `python main.py preview-srt <name>` |
| `tts.py` | `python main.py tts <name>` |
| `make.py` | `python main.py build <name>` |

把旧数据迁过来：

```bash
python main.py init old_project --video "原视频路径"
mv 旧的narration.txt projects/old_project/narration.txt
mv 旧的narration_aligned.txt projects/old_project/narration_aligned.txt
python main.py all old_project
```

## 开发

### 添加新命令

1. 在 `vi/commands/` 建 `<name>.py`，导出 `configure(parser)` 和 `run(args)`
2. 在 `vi/cli.py` 的 `lazy_map` 加路由
3. 算法实现放 `vi/core/`，命令文件只做参数解析 + 调用核心

### 换 ASR 引擎

实现一个新的 `vi/core/whisper_<engine>.py`，签名匹配 `whisper_py.py`，然后在 `commands/transcribe.py` 加分支即可。
