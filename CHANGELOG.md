# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Add `licc` as an alias for `lessons-in-cast`, and expose `audition`,
  `voice-design`, and `effects` as subcommands while preserving standalone entry points.
- Apply the same final audio renderer to auditions, including keyframe-only takes.
- Publish completed audition takes atomically, retry failed takes, and use process-held locks.
- Scope direction validation to effective request context and stop hashing all Core source to resume auditions.
- Share IndexTTS profile parameter assembly with audition tools; bind fixed-path audio reuse to the actual request and reference contents.
- Preserve production input files across generated-output cleanup, including on failure.
- Keep audition backend caches persistent across conversion and post-processing retries.
- Delegate raw-audio cache validation to backends; fail incomplete synthesis and block invalid bundles.
- Pass the configured workspace root to production keyframe processing.
- Exclude transport timeouts and unused character definitions from new API checkpoint bindings.
- Reuse workspace path resolution so venv interpreter symlinks remain supported.
- Avoid hashing unused references or hashing the same reference twice for an IndexTTS cache lookup.
- Back up only inputs covered by the generated-output cleanup list.
- Remove automatic migrations for earlier local pre-release cache formats; retain current-format resume behavior.
- Use version `0.0.2.dev3` for the local pre-release containing these changes.

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
