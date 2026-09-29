# Lessons in Cast Core

[![CI](https://github.com/5o1/Lessons-in-Cast-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/5o1/Lessons-in-Cast-Core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Lessons in Cast is a backend-agnostic dialogue-to-voice pipeline for visual
novels. It extracts dialogue, applies deterministic and AI-assisted
annotations, drives replaceable speech backends, applies audio effects, and
packages the resulting audio for a game release.

```text
galgame source -> backend extraction -> stream processing -> annotation
              -> speech synthesis -> audio integration
```

This repository contains the Python package and its unit tests. Project data —
character configs, voice profiles, prompts and Kantoku direction — lives in a
separate workspace, which `lessons-in-cast init` creates.

## Installation

```bash
pip install git+https://github.com/5o1/Lessons-in-Cast-Core.git
```

Python 3.11 or newer on Linux or Windows. FFmpeg 6 or newer (with libopus for
Opus output) must be on `PATH`; the effects chain uses `alimiter` options that
FFmpeg 4.x does not support. Speech backends (IndexTTS, VoxCPM, MiniMax H3)
run in their own environments and are not installed here.

## Workspace

Create a workspace in the current directory:

```bash
mkdir my-voice-project && cd my-voice-project
lessons-in-cast init --release game_releases/MyGame-1.0
lessons-in-cast check-config
```

`init` writes template configuration (`configs/`), annotation prompts
(`prompts/`), and guides for voice profiles (`profiles/`) and Kantoku direction
(`kantoku/`). Existing files are kept; pass `--force` to overwrite them with
the templates. The generated `README.md` lists the next steps.

Commands operate on a workspace root, located by searching upward from the
current directory for `configs/workspace.toml` (or passed explicitly with
`--root` / `repository_root=`). Generated artifacts go under `build/`.

## Development

```bash
conda env create -f environment.yml   # Python 3.12, FFmpeg >= 6, editable install
conda activate lessons-in-cast-core
pytest
```

CI runs the test suite on Linux and Windows with Python 3.11–3.13, and builds
and smoke-tests the package. Pushing a `v*` tag that matches the version in
`pyproject.toml` creates a GitHub release.

## License

[MIT](LICENSE)
