# Lessons in Cast workspace

A production workspace for [Lessons in Cast Core](https://github.com/5o1/Lessons-in-Cast-Core).
Commands run from this directory or any subdirectory; the workspace root is the
nearest directory containing `configs/workspace.toml`.

```text
configs/    pipeline, character, source, effect and model configuration
profiles/   voice profiles (Python entrypoints and their settings)
prompts/    annotation prompts for the cleaning and polish stages
kantoku/    optional human direction per source file, label and scene
build/      generated run artifacts (ignored by Git)
```

## Getting started

1. Unpack the game release and point `current_game_release` in
   `configs/workspace.toml` at it.
2. List the dialogue-bearing scripts in `configs/dialogue_sources.toml` and the
   game's speakers in `configs/characters.toml`.
3. Register local models in `configs/model_sources.toml` (machine-specific and
   ignored by Git; `configs/model_sources.default.toml` documents the format).
4. Add a voice profile under `profiles/` and select it with a character's
   `default_voice_profile`.
5. Check the configuration, then run the pipeline:

```bash
lessons-in-cast check-config
lessons-in-cast extract
lessons-in-cast prepare
lessons-in-cast annotate            # cleaning
lessons-in-cast validate
lessons-in-cast polish-prepare
lessons-in-cast annotate --stage polish
lessons-in-cast polish-validate
lessons-in-cast plan-tts
lessons-in-cast synthesize
lessons-in-cast bundle
```

With the default `codex` annotation backend, `annotate` exports a task for a
separate Codex session; use `codex-next`, `codex-import` and `codex-status` to
exchange packets. Set a stage's backend to `api` in `configs/pipeline.toml` to
call an OpenAI-compatible API instead.

FFmpeg 6 or newer must be on `PATH`. Speech backends run in their own
environments; see each backend's profile configuration.
