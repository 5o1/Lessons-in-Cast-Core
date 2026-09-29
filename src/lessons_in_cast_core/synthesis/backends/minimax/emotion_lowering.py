"""Explicit, auditable fallbacks for descriptions that T2A cannot accept.

These are backend compilation rules, not core emotion presets or automatic
natural-language understanding. Unknown descriptions deliberately fail closed.
"""

from copy import deepcopy
from dataclasses import replace
import json
import math

from ....performance import AdaptationFidelity, FeatureAdaptation, VocalMode


_EMOTIONS = {"neutral", "calm", "happy", "sad", "angry", "afraid", "disgusted", "surprised"}
_RANGES = {
    "energy": (-1, 1),
    "breathiness": (0, 1),
    "speed": (0.5, 2),
    "volume_gain_db": (-12, 12),
}


def validate_rules(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("MiniMax arbitrary_emotions must be a description-to-rule table")
    for description, rule in value.items():
        if not isinstance(description, str) or not description.strip():
            raise ValueError("MiniMax emotion descriptions cannot be empty")
        if not isinstance(rule, dict) or set(rule) - {"emotion", "vocal_mode", *_RANGES}:
            raise ValueError(f"Invalid MiniMax emotion rule: {description!r}")
        if not isinstance(rule.get("emotion"), str) or rule["emotion"] not in _EMOTIONS:
            raise ValueError(f"Unsupported MiniMax fallback emotion: {description!r}")
        if "vocal_mode" in rule and rule["vocal_mode"] not in {"normal", "whisper", "shout"}:
            raise ValueError("MiniMax fallback vocal_mode must be normal, whisper or shout")
        for name, (low, high) in _RANGES.items():
            if name in rule:
                number = rule[name]
                if (isinstance(number, bool) or not isinstance(number, (int, float))
                        or not math.isfinite(number) or not low <= number <= high):
                    raise ValueError(f"MiniMax fallback {name} must be {low}..{high}")
    return deepcopy(value)


def lower_arbitrary_job(job, rules):
    description = job.arbitrary_emotion
    if description not in rules:
        raise ValueError(f"MiniMax has no explicit arbitrary_emotion fallback for {description!r}")
    rule = rules[description]
    # Explicit core controls take precedence over the description's approximation.
    changes = {name: (VocalMode(value) if name == "vocal_mode" else value)
               for name, value in rule.items()
               if name != "emotion" and getattr(job.performance, name) is None}
    lowered = replace(job, arbitrary_emotion=None, emotion=rule["emotion"],
                      performance=replace(job.performance, **changes))
    note = FeatureAdaptation(
        "arbitrary_emotion", AdaptationFidelity.APPROXIMATED,
        "Profile-authored fallback; MiniMax T2A does not interpret free-form direction. "
        + json.dumps({"description": description, "rule": rule}, ensure_ascii=False)
        + "; explicit core performance controls take precedence. "
        "Mixed emotions, intensity and vocal instability are not reproduced exactly.",
    )
    return lowered, note
