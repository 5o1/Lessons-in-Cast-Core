"""Strict local QwenEmotion inference shared by preset and arbitrary emotions."""

import math
from pathlib import Path

PREDICTOR_VERSION = 1
MAX_NEW_TOKENS = 512


def load_qwen_emotion(model_path):
    """Load only the official emotion model, rejecting silent neutral fallbacks."""
    import torch
    from indextts.infer_v2_5 import QwenEmotion
    from .emotion_preparation import current_file_hash

    model_path = Path(model_path)
    for weight in model_path.glob("*.safetensors"):
        metadata = model_path.parent / ".cache/huggingface/download" / model_path.name / (weight.name + ".metadata")
        if metadata.exists():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2 and len(lines[1]) == 64 and current_file_hash(weight) != lines[1]:
                raise ValueError(f"QwenEmotion weight checksum differs from its download record: {weight}")

    torch.set_num_threads(4)
    engine = QwenEmotion(str(model_path))
    if not torch.cuda.is_available():
        engine.model.float()
    engine.model.eval()
    engine.raw_output = None
    generate = engine.model.generate
    convert = engine.convert

    def bounded_generate(**kwargs):
        kwargs.update(do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        with torch.inference_mode():
            result = generate(**kwargs)
        output = result[0][kwargs["input_ids"].shape[-1]:]
        engine.raw_output = engine.tokenizer.decode(output, skip_special_tokens=True)
        if len(output) >= MAX_NEW_TOKENS:
            raise ValueError("QwenEmotion reached its token limit; refusing to cache a truncated prediction")
        return result

    def strict_convert(content):
        normalized = engine.normalize_content(content)
        present = {key: normalized[key] for key in engine.desired_vector_order if key in normalized}
        # The upstream melancholic workaround inserts zero sad/melancholic values
        # even for an unparseable response; require an actual nonzero detection.
        if not present or not any(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                                  for v in present.values()):
            raise ValueError(f"QwenEmotion produced no valid emotion scores; refusing neutral fallback. Raw output: {engine.raw_output!r}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not 0 <= v <= 1.2 for v in present.values()):
            raise ValueError(f"Invalid QwenEmotion scores: {present!r}")
        return convert(normalized)

    engine.model.generate = bounded_generate
    engine.convert = strict_convert
    return engine


def resolve_arbitrary_emotion(description, engine, energy=None):
    """Runtime-only conversion; does not register or write a preset cache."""
    from .emotion_preparation import validate_prediction
    from .adapter import normalize_index_emotion_vector

    if not isinstance(description, str) or not description.strip():
        raise ValueError("arbitrary_emotion requires a nonempty description")
    prediction = engine.inference(description)
    vector = normalize_index_emotion_vector(validate_prediction(prediction))
    if energy is not None:
        vector = [v * (1 + energy * .5) for v in vector]
        total = sum(vector)
        if total > .8:
            vector = [v * .8 / total for v in vector]
        vector = [round(v, 6) for v in vector]
    return {"prediction": prediction, "vector": vector, "raw_output": getattr(engine, "raw_output", None),
            "predictor_version": PREDICTOR_VERSION}
