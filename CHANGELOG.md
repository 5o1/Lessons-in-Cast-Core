# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.0.1] - 2026-09-29

First public release of the core package.

### Added

- Dialogue pipeline: Ren'Py extraction, batching, scoped runs, annotation
  (Codex packet exchange or an OpenAI-compatible API) with validation and
  human overrides, a separate polish pass, synthesis planning, rendering and
  release bundling.
- Voice profiles loaded from the workspace, with IndexTTS, MiniMax Speech,
  MiniMax H3 and GPT-SoVITS backends.
- FFmpeg effects chain, keyframe gain envelopes, pronunciation lexicon,
  Kantoku direction, character auditions and VoxCPM2 voice design.
- `lessons-in-cast init` creates a workspace with template configuration and
  prompts in the current directory.
- Support for Linux and Windows.

[0.0.1]: https://github.com/5o1/Lessons-in-Cast-Core/releases/tag/v0.0.1
