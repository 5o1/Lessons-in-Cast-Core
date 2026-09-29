# Voice profiles

A profile is a Python entrypoint that builds a backend-neutral `VoicePipeline`.
Characters select one through `default_voice_profile` in
`configs/characters.toml`; entrypoints must live inside this directory.

Profiles can use either layout:

```text
profiles/
├── simple_profile.py
└── example_index_tts/
    ├── pipeline.py
    └── config.toml
```

A single-file profile may keep all settings in Python. A bundle profile owns
its configuration and resources beside its entrypoint;
`context.resolve_resource(...)` resolves bundle-relative paths and rejects
paths that escape the bundle.

Every entrypoint must export:

```python
def create_pipeline(context: VoiceProfileContext) -> VoicePipeline:
    ...
```

For example, a bundle built on the IndexTTS backend:

```python
from lessons_in_cast_core.synthesis.backends.index_tts import IndexTtsPipeline
from lessons_in_cast_core.synthesis.profiles import VoicePipeline, VoiceProfileContext


class ExampleIndexTtsPipeline(IndexTtsPipeline):
    @property
    def pipeline_id(self) -> str:
        return "example-index-tts"

    @property
    def character_id(self) -> str:
        return "a"


def create_pipeline(context: VoiceProfileContext) -> VoicePipeline:
    return ExampleIndexTtsPipeline(context)
```

Models are referenced by ID from `configs/model_sources.toml`, never by local
path. Reference audio belongs in a bundle's `assets/` directory, which is
ignored by Git.

Prepare every configured reference with `lessons-in-cast prepare-voices`;
synthesis also prepares them lazily.
