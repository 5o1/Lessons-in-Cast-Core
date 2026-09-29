"""Offline QwenEmotion compilation with resumable, provenance-bound caches."""

from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ....emotion_presets import EmotionPresetPreprocessor, PresetCatalog
from ....emotions import emotion_definitions
from ....paths import SOURCE_ROOT
from .adapter import normalize_index_emotion_vector

AXES = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
VERSION = 1


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=128)
def _cached_file_hash(path: Path, size: int, modified: int, inode: int) -> str:
    return file_hash(path)


def current_file_hash(path: Path) -> str:
    stat = path.stat()
    return _cached_file_hash(path.resolve(), stat.st_size, stat.st_mtime_ns, stat.st_ino)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def input_text(definition, kind: str) -> str:
    if kind == "description":
        return definition["description"]
    if kind == "example":
        return definition["example"]
    if kind == "combined":
        return f"Emotion: {definition['description']}\nDialogue: {definition['example']}"
    raise ValueError(f"Unknown emotion input kind: {kind}")


def validate_prediction(prediction) -> list[float]:
    if not isinstance(prediction, dict) or set(prediction) != set(AXES):
        raise ValueError("QwenEmotion must return exactly eight named axes")
    values = [prediction[key] for key in AXES]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1.2 for v in values):
        raise ValueError("QwenEmotion scores must be finite numbers between 0 and 1.2")
    return values


def validate_entry(entry, definition, context) -> list[float]:
    text = input_text(definition, context["input_kind"])
    if entry.get("definition") != definition or entry.get("input") != text:
        raise ValueError("Emotion preset changed; rerun emotion-presets prepare")
    expected_key = digest({"context": context, "definition": definition, "input": text})
    if entry.get("key") != expected_key:
        raise ValueError("Emotion cache provenance mismatch")
    raw = validate_prediction(entry["prediction"])
    if entry.get("vector") != normalize_index_emotion_vector(raw):
        raise ValueError("Emotion cache vector does not match its prediction")
    return raw


def load_vector_cache(path: Path, catalog: PresetCatalog, *, model_id: str | None = None,
                      model_definition=None, source_root: Path | None = None):
    """Fail closed on missing, incomplete, stale or corrupt generated mappings."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing emotion cache: {path}; run emotion-presets prepare --backend index-tts")
    document = json.loads(path.read_text(encoding="utf-8"))
    context = document["context"]
    if (document.get("version") != VERSION or document.get("complete") is not True
            or context.get("backend") != "index-tts" or document.get("axes") != list(AXES)):
        raise ValueError("Invalid or incomplete IndexTTS emotion cache")
    if model_id is not None and context["model"]["id"] != model_id:
        raise ValueError("Emotion cache belongs to a different model ID")
    if model_definition is not None and context["model"] != model_definition:
        raise ValueError("Registered emotion model metadata changed; rerun emotion-presets prepare")
    if source_root is not None and context.get("implementation_sha256") != current_file_hash(source_root / "indextts/infer_v2_5.py"):
        raise ValueError("QwenEmotion implementation changed; rerun emotion-presets prepare")
    if "predictor_sha256" in context and context["predictor_sha256"] != current_file_hash(Path(__file__).with_name("qwen_emotion.py")):
        raise ValueError("QwenEmotion predictor changed; rerun emotion-presets prepare")
    definitions = emotion_definitions(catalog, catalog)
    if set(document["entries"]) != set(definitions):
        raise ValueError("Emotion catalog changed; rerun emotion-presets prepare")
    vectors = {name: validate_entry(document["entries"][name], definition, context)
               for name, definition in definitions.items()}
    return vectors, digest(document)


def compile_cache(definitions, context, output: Path, engine_factory, *, force=False) -> Path:
    """Load the model lazily, predict only uncached labels, checkpoint each one."""
    partial = output.with_suffix(output.suffix + ".partial")
    entries = {}
    if not force:
        for candidate in (output, partial):
            if candidate.exists():
                try:
                    previous = json.loads(candidate.read_text(encoding="utf-8"))
                    if previous.get("context") == context:
                        entries.update(previous.get("entries", {}))
                except (ValueError, TypeError):
                    pass
    document = {"version": VERSION, "complete": False, "axes": list(AXES),
                "context": context, "entries": {}}
    engine = None
    for index, (name, definition) in enumerate(sorted(definitions.items()), 1):
        entry = entries.get(name)
        try:
            validate_entry(entry, definition, context)
            reused = True
        except (ValueError, KeyError, TypeError, AttributeError):
            if engine is None:
                engine = engine_factory()
            text = input_text(definition, context["input_kind"])
            try:
                prediction = engine.inference(text)
            except Exception as exc:
                atomic_json(output.with_suffix(output.suffix + ".failure.json"),
                            {"label": name, "input": text, "context": context, "error": str(exc),
                             "raw_output": getattr(engine, "raw_output", None)})
                raise
            raw = validate_prediction(prediction)
            entry = {"definition": definition, "input": text, "prediction": prediction,
                     "raw_output": getattr(engine, "raw_output", None),
                     "vector": normalize_index_emotion_vector(raw),
                     "key": digest({"context": context, "definition": definition, "input": text})}
            reused = False
        document["entries"][name] = entry
        atomic_json(partial, document)
        print(f"[{index}/{len(definitions)}] {name}: {'cached' if reused else 'generated'} {entry['vector']}", flush=True)
    groups = {}
    for name, entry in document["entries"].items():
        groups.setdefault(tuple(entry["vector"]), []).append(name)
    document["warnings"] = [f"Identical vectors: {', '.join(names)}" for names in groups.values() if len(names) > 1]
    for warning in document["warnings"]:
        print(f"WARNING: {warning}", flush=True)
    document["complete"] = True
    atomic_json(output, document)
    partial.unlink(missing_ok=True)
    output.with_suffix(output.suffix + ".failure.json").unlink(missing_ok=True)
    return output


class IndexTtsEmotionPreprocessor(EmotionPresetPreprocessor):
    backend = "index-tts"

    def __init__(self, root, python, source, model_path, model_definition, input_kind="description"):
        self.root, self.python, self.source = root, python, source
        self.model_path, self.model_definition = model_path, model_definition
        self.input_kind = input_kind

    @classmethod
    def from_profile(cls, root: Path, profile: Path, input_kind="description"):
        from .config import load_index_tts_pipeline_config
        from ....model_registry import load_model_registry
        config = load_index_tts_pipeline_config(profile, repository_root=root)
        registry = load_model_registry(repository_root=root)
        return cls(root, (root / config.python_executable).absolute(),
                   (root / config.source_root).resolve(), registry.resolve_path(config.model_id),
                   registry.require(config.model_id).to_dict(), input_kind)

    def prepare(self, catalog: PresetCatalog, output: Path, *, force: bool = False) -> Path:
        # The official config is read in the backend environment (PyYAML installed
        # there); the control interpreter needs only the standard library.
        request = {"definitions": emotion_definitions(catalog, catalog), "source": str(self.source),
                   "model_path": str(self.model_path), "model": self.model_definition,
                   "input_kind": self.input_kind, "output": str(output.resolve()), "force": force}
        request_path = output.with_suffix(output.suffix + ".request.json")
        atomic_json(request_path, request)
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        environment["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), str(self.source)))
        environment["MPLCONFIGDIR"] = str(self.root / "build/cache/matplotlib")
        subprocess.run([str(self.python), "-m", __name__, str(request_path.resolve())],
                       cwd=self.root, env=environment, check=True)
        load_vector_cache(output, catalog, model_id=self.model_definition["id"])
        return output


def worker(request_path: Path) -> None:
    import yaml
    request = json.loads(request_path.read_text(encoding="utf-8"))
    model_path, source = Path(request["model_path"]), Path(request["source"])
    config_path = model_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    emotion_path = model_path / config["qwen_emo_path"]
    weights = sorted(emotion_path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No QwenEmotion weights in {emotion_path}")
    print("Fingerprinting local QwenEmotion model and implementation...", flush=True)
    files = sorted(set(weights + list(emotion_path.glob("*.json")) + list(emotion_path.glob("*.txt")) + list(emotion_path.glob("*.jinja"))))
    context = {"backend": "index-tts", "compiler_version": VERSION, "model": request["model"],
               "input_kind": request["input_kind"], "generation": {"do_sample": False, "max_new_tokens": 512},
               "model_files": {p.name: file_hash(p) for p in files},
               "config_sha256": file_hash(config_path),
               "predictor_sha256": file_hash(Path(__file__).with_name("qwen_emotion.py")),
               "implementation_sha256": file_hash(source / "indextts/infer_v2_5.py")}

    def engine_factory():
        from .qwen_emotion import load_qwen_emotion
        print("Loading local QwenEmotion (no speech model)...", flush=True)
        return load_qwen_emotion(emotion_path)

    compile_cache(request["definitions"], context, Path(request["output"]), engine_factory, force=request["force"])


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
