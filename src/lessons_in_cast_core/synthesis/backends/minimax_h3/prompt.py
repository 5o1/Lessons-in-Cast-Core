"""Lower portable acting intent to H3's six-section reference prompt."""

import json
from pathlib import Path

from ....performance import AdaptationFidelity, FeatureAdaptation, SpeechAdaptation, resolve_legacy_delivery
from ....pronunciations import pronunciation_matches
from ....speech_markup import SpeechSegment
from ...references.voices import resolve_voice_reference
from .config import MiniMaxH3Config


def compile_prompt(job, config: MiniMaxH3Config, reference: Path, catalog, pronunciations=()):
    if not job.text.strip():
        raise ValueError("H3 requires nonempty cleaned dialogue")
    segments = job.segments or (SpeechSegment(job.text, None if job.arbitrary_emotion else job.emotion or "neutral",
                                             job.voice, job.arbitrary_emotion),)
    if "".join(segment.text for segment in segments) != job.text:
        raise ValueError("H3 segments must preserve the complete job text")
    references = []
    directions = []
    features = [FeatureAdaptation("reference_voice", AdaptationFidelity.APPROXIMATED,
                                  "H3 Ref2VA audio conditioning; identity preservation is not guaranteed")]
    for segment in segments:
        if segment.emotion and segment.arbitrary_emotion:
            raise ValueError("Preset and arbitrary emotions cannot overlap")
        path = resolve_voice_reference(reference, segment.voice) if segment.voice else reference.resolve()
        if path not in references:
            references.append(path)
        index = references.index(path) + 1
        description = segment.arbitrary_emotion
        if segment.emotion:
            if segment.emotion not in catalog:
                raise ValueError(f"Unknown emotion preset: {segment.emotion}")
            description = catalog[segment.emotion][0]
        # Dialogue delimiters are protocol tokens, not an escaping convention.
        if "<" in segment.text or ">" in segment.text:
            raise ValueError("Clean angle-bracket markup before H3 synthesis")
        directions.append(f"(S1), using the voice timbre of <Audio {index}>, "
                          f"{description or 'speaks naturally without exaggerated emotion'}. "
                          f"Says exactly: <d>[{config.language}]{segment.text}</d>")
        if description:
            features.append(FeatureAdaptation("arbitrary_emotion" if segment.arbitrary_emotion else "emotion",
                                              AdaptationFidelity.APPROXIMATED,
                                              "Natural-language acting direction, not a calibrated emotion vector"))
        if segment.voice:
            features.append(FeatureAdaptation("voice", AdaptationFidelity.APPROXIMATED,
                                              f"Reference {index} selected from {path.name}"))
    if len(references) > 3:
        raise ValueError("H3 supports at most three reference audio files per take")

    performance, legacy_features = resolve_legacy_delivery(job.performance, job.delivery)
    features.extend(legacy_features)
    controls = [config.direction]
    if performance.direction:
        controls.append(performance.direction)
    if performance.vocal_mode:
        controls.append(f"Vocal mode: {performance.vocal_mode.value}.")
    speed = config.base_speed * (performance.speed if performance.speed is not None else 1)
    controls.append(f"Speaking rate: {speed:g} times a natural conversational pace.")
    for field in ("pitch_semitones", "volume_gain_db", "energy", "brightness", "clarity", "breathiness"):
        value = getattr(performance, field)
        if value is not None:
            controls.append(f"Requested {field}: {value:g} (relative to the reference voice).")
    for cue in performance.cues:
        if not 0 <= cue.offset <= len(job.text):
            raise ValueError("Performance cue offset exceeds the H3 dialogue")
        controls.append(f"At the boundary after {json.dumps(job.text[:cue.offset])}, before "
                        f"{json.dumps(job.text[cue.offset:])}, perform {cue.kind.value}"
                        + (f" for {cue.duration_seconds:g} seconds" if cue.duration_seconds is not None else "")
                        + (f" with intensity {cue.intensity:g}" if cue.intensity is not None else "") + ".")
    features.append(FeatureAdaptation("performance", AdaptationFidelity.APPROXIMATED,
                                      "Prompt directions; numeric speed, pitch, gain and cue timing are not exact controls"))
    for rule in pronunciations:
        if pronunciation_matches(job.text, rule):
            controls.append(f"Pronounce {json.dumps(rule.term)} as {json.dumps(rule.pronunciation)} "
                            f"({rule.system}). Lexical prosody intent: "
                            f"{json.dumps(rule.prosody.to_dict(), ensure_ascii=False)}. Do not read these instructions aloud.")
            features.append(FeatureAdaptation("pronunciation", AdaptationFidelity.APPROXIMATED,
                                              f"Prompt-only {rule.system} and lexical prosody for {rule.term}"))
    sections = {
        "subject_definitions": "\n".join(f"<Audio {i}> is the voice-timbre reference for the sole speaker (S1)."
                                           for i in range(1, len(references) + 1)),
        "summary": "[reference generation + audio reference] A single-speaker dialogue recording. "
                   "Reference only vocal identity; speak new dialogue in the requested language.",
        "retention_analysis": "\n".join(f"<Audio {i}>: reference - retain speaker identity and timbre, "
                                          "not the original words, background sound or emotional delivery."
                                          for i in range(1, len(references) + 1)),
        "detailed_description": "[Shot 1] A static neutral background. One uninterrupted voice recording. "
                                "Start the requested dialogue promptly. Speak every supplied word once, in order. "
                                "No introductory words, no extra speech, no singing or humming unless directed. "
                                "Finish all words naturally, then remain silent.\n"
                                + " ".join(controls) + "\n" + "\nThen ".join(directions),
        "overall_soundscape": "Clean close-microphone dry voice only. No ambience, music, sound effects or other speakers.",
        "non_diegetic_music": "None.",
    }
    prompt = "\n\n".join(f"{key}:\n{value}" for key, value in sections.items())
    return SpeechAdaptation(job.id, job.dialogue_id, "minimax-h3", job.text, job.emotion,
                            {"prompt": prompt, "references": [str(p) for p in references],
                             "width": 32, "height": 32, "length": config.frame_count,
                             "duration_seconds": config.frame_count / 24,
                             "steps": config.steps, "seed": config.seed,
                             "audio_only_decode": True}, tuple(features))
