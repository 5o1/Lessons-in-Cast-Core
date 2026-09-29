# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`lessons_in_cast_core` is the Python library behind Lessons in Cast. It is a pipeline that turns visual-novel dialogue into voice lines and doesn't depend on any one speech backend. This repo holds **only the package and its unit tests**. All project data lives in a separate *workspace* (created by `lessons-in-cast init`): character configs, voice profiles, prompts, Kantoku direction and release scripts. Don't add workspace data or references to any particular game or private workspace here.

## Commands

```bash
conda env create -f environment.yml      # Python 3.12, FFmpeg >= 6, editable install with [dev,reference]
conda activate lessons-in-cast-core
pytest                                   # full suite (tests/)
pytest tests/test_effects.py             # one file
pytest tests/test_effects.py::EffectsTests::test_fades_preserve_duration_and_other_samples
```

- The base `python` has no pytest. Activate the env first, or use `conda run -n lessons-in-cast-core pytest`.
- Tests are `unittest.TestCase` classes (they use `subTest`) and run under pytest. Test doubles are in `tests/fakes.py`. Record/annotation builders and `write_minimal_workspace()` are in `tests/helpers.py`.
- FFmpeg >= 6 must be on `PATH`. The effects chain uses `alimiter` options that FFmpeg 4.x lacks.
- The project has no linter or formatter config.
- CI (`.github/workflows/ci.yml`) runs the suite on Ubuntu and Windows with Python 3.11–3.13, then builds the wheel, installs it into a clean venv and runs `lessons-in-cast init` followed by `check-config`. `release.yml` runs when a `v*` tag is pushed. The tag must match `project.version`. PyPI publishing only runs when the repo variable `PUBLISH_TO_PYPI` is `true`.

Console scripts (defined in `pyproject.toml`): `lessons-in-cast` (`cli.py`), `lessons-in-cast-voice-design`, `lessons-in-cast-audition`, `lessons-in-cast-effects`.

## Workspace contract

Every command runs against a workspace root. `config.find_repository_root()` finds it by walking up from the CWD until it reaches `configs/workspace.toml`. You can also pass it with `--root` or `repository_root=`. The workspace provides `configs/`, `profiles/`, `prompts/` and `kantoku/`. Run artifacts go to `build/current/` by default (`--build-dir`). `ArtifactLayout` in `workflow/artifacts.py` is the single source of truth for artifact file names (`dialogue.tab`, `raw.jsonl`, `annotation_requests.jsonl`, `validated.jsonl`, `polish/`, …).

`lessons-in-cast init` (`workspace_init.py`) creates a new workspace in the CWD (or `--root`) from `src/lessons_in_cast_core/templates/workspace/`. These files ship as package data. `gitignore` is renamed to `.gitignore`, and `configs/model_sources.toml` is seeded from the default. Existing files are kept unless `--force` is passed. When you add a required workspace file or config key, update the templates too. `tests/test_workspace_init.py` checks that a freshly initialized workspace passes `check-config`.

`paths.py` locates the package from its own `__file__`, not from `<workspace>/src`. Subprocess workers that need to import the package (for example the IndexTTS emotion preparation) get `SOURCE_ROOT` added to `PYTHONPATH`.

## Pipeline architecture

```
galgame source -> extract -> prepare (raw JSONL + batched annotation requests)
  -> annotate "cleaning" -> validate (+ overrides)
  -> polish-prepare -> annotate "polish" -> polish-validate
  -> plan-tts -> synthesize -> effects/render -> bundle
```

`DialoguePipeline` in `pipeline.py` orchestrates the model-neutral stages. `cli.py` maps each subcommand onto it. `run-production` chains validation, synthesis and bundling. Every stage records hashes and provenance in a run manifest (`workflow/manifest.py`), and caching is driven by content hashes (`hashing.py`).

The seams you can swap out:

- **Galgame backends** (`galgame/`): implement `GalgameBackend` (`galgame/api.py`) and register it in `_BACKEND_FACTORIES` in `galgame/loader.py`. Ren'Py is the only one so far (`galgame/renpy/`, which covers archive extraction, dialogue reading and integration into the game). The configured ID has to match the injected backend.
- **Annotation** (`annotation/`): two stages, `cleaning` and `polish`. Each runs through either an API backend (`annotation/api.py`, OpenAI-compatible, with the key read from an env var named in config) or the manual **Codex** packet exchange (`codex-next` / `codex-import` / `codex-status` in `annotation/codex.py`). Responses must match `annotation/schema.py` and are checked by `AnnotationValidator`. Human overrides from `annotation/overrides.py` are applied after that.
- **Voice profiles** (`synthesis/profiles/`): these are *user Python files* under the workspace's `profiles/` directory. `load_voice_profile` imports them dynamically, refuses any path outside `profiles/`, and calls their `create_pipeline(context)`, which must return a `VoicePipeline` (`synthesis/profiles/api.py`). `configuration` must include every setting that can change the output, because it feeds the cache fingerprints. Profiles that don't implement inline segments, voice tags or reference overrides must raise `NotImplementedError` and must not silently ignore them.
- **Speech backends** (`synthesis/backends/`): IndexTTS, MiniMax (HTTP), MiniMax H3, GPT-SoVITS. The heavy backends (IndexTTS, VoxCPM, MiniMax H3) run **in their own environments** as subprocesses. For example, `index_tts/worker.py` is a JSON-lines worker that runs inside the official IndexTTS env. Don't import their dependencies into core modules. The `reference` extra (librosa/numpy/soundfile) is imported lazily inside functions only.
- **Effects** (`effects/`): an FFmpeg-based effect catalog and processor, used by `WaveRenderer` (`synthesis/audio.py`).

Other cross-cutting modules:

- `performance.py` and `speech_markup.py` hold the backend-neutral performance, cue and segment model. Each backend lowers it into its own parameters, e.g. `minimax/emotion_lowering.py` and `index_tts/emotions.py`.
- `kantoku.py` resolves human direction TOML (per source file, label or scene) and fingerprints it.
- `pronunciations.py`, `emotions.py` and `emotion_presets.py` provide shared text and emotion vocabulary. The allowed emotions and effects are set in the config.
- **Linux and Windows are both supported.** Always pass `encoding="utf-8"` to text file I/O, and pass `encoding="utf-8", errors="replace"` to `subprocess` calls that use `text=True`. Use `locking.exclusive_lock` for file locks, not `fcntl`, and `paths.venv_python` for interpreters inside a venv. Ren'Py extraction runs `*.sh` on Linux; on Windows it runs `<Game>.py` with the bundled `lib/*windows*/python.exe`.
- `config.py` defines frozen dataclasses for all workspace TOML and raises `ConfigurationError` when something is inconsistent. `check-config` validates the whole workspace.
