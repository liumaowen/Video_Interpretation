# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Video_Interpretation（简称 vi）是一个 Python CLI 工具，用于将电影预告片自动合成为 B 站影评/解说视频。给定一个原版预告片 + 一段中文解说稿，它会输出一个带配音、背景音乐和双字幕（原英文 + 中文解说）的成品 mp4。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 CLI
python main.py <command> [project_name] [options]

# 列出所有项目
python main.py ls [-l]

# 一键跑完全流程
python main.py all <name>

# 各个步骤（完整流程：transcribe → refine → tts → build）
python main.py transcribe [name]   # ASR：通过 faster-whisper 识别英文字幕
python main.py refine [name]       # 重新切分 SRT，避免一行过长
python main.py llm-narrate [name]  # 通过 Claude API 生成中文解说稿
python main.py align [name]        # 为解说稿标注时间轴
python main.py tts [name]          # 通过 edge-tts 合成配音
python main.py build [name]        # 混音 + 烧入双字幕
python main.py preview-srt [name]  # 预览字幕节奏（不跑 TTS）
python main.py clean [name]        # 清理中间产物
```

## 架构

### 入口
- [main.py](main.py) — 设置 `HF_HOME` 环境变量，委托给 `vi.cli:main`

### 包结构（`vi/`）

```
vi/
├── cli.py              # argparse 路由，重命令懒加载
├── config.py           # 分层配置：CLI 参数 > project.toml > config.toml
├── paths.py            # 路径解析，支持从 cwd 自动发现项目
├── commands/           # 子命令实现（每个模块导出 configure/run）
│   ├── init.py         # 创建项目骨架
│   ├── ls.py           # 列出项目
│   ├── transcribe.py   # ASR 转录
│   ├── refine.py       # SRT 重新切分
│   ├── llm_narrate.py  # LLM 解说稿生成
│   ├── align.py        # 时间轴对齐
│   ├── preview_srt.py  # SRT 预览（不跑 TTS）
│   ├── tts.py          # 文本转语音
│   ├── build.py        # 最终视频合成
│   ├── all_cmd.py      # 流程编排器
│   └── clean.py        # 清理中间产物
└── core/               # 算法实现
    ├── whisper_py.py   # faster-whisper Python 引擎
    ├── whisper_exe.py  # whisper-faster.exe 引擎
    ├── refine.py       # SRT 精炼算法
    ├── tts_engine.py   # edge-tts 封装，支持自动加速
    ├── ffmpeg.py       # ffmpeg 字幕烧录
    ├── llm.py          # Claude API 封装
    ├── llm_prompts.py  # LLM 提示词模板
    ├── srt.py          # SRT 解析/格式化工具
    └── preview_srt.py  # 按字符均分的 SRT 预览
```

### 配置系统
- [vi/config.py](vi/config.py) 通过 `_deep_merge` 实现优先级：CLI 参数 > `project.toml` > `config.toml`
- 全局默认值在 [config.toml](config.toml)；每个项目可在 `projects/<name>/project.toml` 中覆盖

### 路径解析
- [vi/paths.py](vi/paths.py) — 所有命令使用绝对路径；`find_project_root()` 从 cwd 向上查找 `project.toml` 实现自动项目发现

### 懒加载
重量级命令（依赖 whisper/moviepy）在 [vi/cli.py](vi/cli.py) 中懒加载，避免 `ls` 等简单命令启动缓慢。

### 添加新命令
1. 在 `vi/commands/<name>.py` 创建文件，导出 `configure(parser)` 和 `run(args)`
2. 在 `vi/cli.py` 的 `lazy_map` 字典中注册
3. 算法逻辑放 `vi/core/`，命令文件只做参数解析 + 调用核心模块

## 依赖
- `faster-whisper` — ASR 转录
- `edge-tts` — 文本转语音（Microsoft Edge 在线 TTS）
- `moviepy` — 音频混合
- `anthropic` — Claude API，用于 LLM 解说稿生成
- 外部：`ffmpeg` 必须在 PATH 中

## 环境变量
- `ANTHROPIC_API_KEY` — `llm-narrate` 命令必需
- `HF_ENDPOINT` — 可选，国内设置为 `https://hf-mirror.com` 可加速 HuggingFace 模型下载

## ModelScope Notebook 部署

ModelScope 提供免费的在线 Notebook 环境（CPU/GPU），适合开发和测试。

### 快速上手

1. 打开 [ModelScope Notebook](https://modelscope.cn/notebook)，创建或进入一个 Notebook
2. 克隆或上传项目代码到 `/root/Video_Interpretation/`
3. 运行安装脚本：

```bash
cd /root/Video_Interpretation
python setup_modelscope.py
```

这会自动安装：
- `ffmpeg`（视频处理必需）
- `fonts-noto-cjk`（中文字幕字体）
- 所有 Python 依赖（`requirements.txt`）

### 注意事项

- **字体**：Windows 默认字体（Arial、Microsoft YaHei）在 Linux 上不存在，代码会自动 fallback 到 DejaVu Sans / Noto Sans CJK SC
- **ASR 速度**：CPU 环境跑 `tiny` 模型处理 2 分钟预告片约 1-5 分钟；GPU 环境快 5-10 倍
- **网络**：edge-tts 和 Anthropic API 需要外网，ModelScope 默认支持
- **上传素材**：通过 Notebook 界面上传预告片视频到 `projects/<name>/` 目录
