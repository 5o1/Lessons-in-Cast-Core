"""Top-level orchestration for the dialogue-to-audio pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .annotation import (
    ValidatedAnnotation,
    build_annotation_request,
)
from .characters import CharacterDefinition
from .config import PipelineConfig
from .dialogue import (
    DialogueBatchBuilder,
    DialogueRecord,
    JsonlDialogueReader,
    JsonlDialogueWriter,
    audit_dialogue,
)
from .hashing import content_hash, file_hash
from .performance import SpeechAdaptation
from .jsonl import AtomicJsonlWriter, JsonlIndex, read_jsonl, write_jsonl
from .galgame import (
    DialogueExtractionRequest,
    GalgameBackend,
    VoiceManifestWriter,
)
from .synthesis import (
    AudioQualityChecker,
    AudioQualityResult,
    RenderTask,
    SpeechSynthesizer,
    SynthesisPlanner,
    SynthesisIssue,
    TtsJob,
    WaveRenderer,
)
from .effects import CoreEffectProcessor
from .synthesis.audio import AudioRenderError
from .workflow import (
    ArtifactLayout,
    PipelineRequest,
    PipelineResult,
    ValidationSummary,
)
from .workflow.manifest import start_run_manifest, update_run_manifest
from .workflow.validation import AnnotationValidationStage


class DialoguePipeline:
    """Coordinate every model-neutral stage and replaceable backend."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        characters: dict[str, CharacterDefinition],
        synthesizer: SpeechSynthesizer | None = None,
        galgame_backend: GalgameBackend | None = None,
        renderer: WaveRenderer | None = None,
    ) -> None:
        self._config = config
        self._characters = characters
        self._synthesizer = synthesizer
        self._galgame_backend = galgame_backend
        if (
            galgame_backend is not None
            and galgame_backend.backend_id != config.galgame.backend
        ):
            raise ValueError(
                "Configured galgame backend ID does not match the injected backend: "
                f"{config.galgame.backend!r} != {galgame_backend.backend_id!r}"
            )
        self._renderer = renderer or WaveRenderer(
            CoreEffectProcessor(config.effects, sample_rate=config.audio.sample_rate,
                                channels=config.audio.channels, ffmpeg_executable=config.audio.ffmpeg_executable),
            audio_config=config.audio,
            repository_root=config.repository_root,
        )
        self._validation = AnnotationValidationStage(config)

    def _require_galgame_backend(self) -> GalgameBackend:
        if self._galgame_backend is None:
            raise RuntimeError("No galgame backend is configured")
        return self._galgame_backend

    def extract(
        self,
        request: DialogueExtractionRequest,
    ) -> Path:
        result = self._require_galgame_backend().extract_dialogue(request)
        layout = ArtifactLayout(request.output_path.resolve().parent)
        update_run_manifest(
            layout,
            {
                "extraction": {
                    "release_path": str(request.release_path.resolve()),
                    "dialogue_path": str(result.dialogue_path.resolve()),
                    "dialogue_sha256": file_hash(result.dialogue_path),
                    "source_count": result.source_count,
                    "command": list(result.command),
                }
            },
        )
        return result.dialogue_path

    def prepare(self, request: PipelineRequest) -> tuple[int, int]:
        """Import dialogue losslessly and export fixed-window requests."""

        layout = ArtifactLayout(request.artifact_root.resolve())
        layout.root.mkdir(parents=True, exist_ok=True)
        dialogue_tab_hash = file_hash(request.dialogue_tab_path)
        start_run_manifest(
            layout,
            dialogue_path=request.dialogue_tab_path,
            dialogue_hash=dialogue_tab_hash,
        )
        source_filter = request.allowed_sources or None
        if request.dialogue_scope is not None:
            scoped_source = Path(request.dialogue_scope.source)
            if source_filter is not None and scoped_source not in source_filter:
                raise ValueError(
                    f"Dialogue scope source is not allowed: {scoped_source}"
                )
            source_filter = (scoped_source,)
        records = self._require_galgame_backend().read_dialogue(
            request.dialogue_tab_path,
            allowed_sources=source_filter,
            source_root=request.source_root,
        )
        if request.dialogue_scope is not None:
            records = iter(request.dialogue_scope.apply(records))
        dialogue_count = JsonlDialogueWriter().write(
            records,
            layout.raw_dialogue,
        )
        batches = DialogueBatchBuilder(self._config.batching).build(
            JsonlDialogueReader().read(layout.raw_dialogue),
            target_characters=(
                request.dialogue_scope.voice_characters
                if request.dialogue_scope is not None
                and request.dialogue_scope.voice_characters
                else None
            ),
        )
        from .kantoku import Kantoku, bind_direction, split_directed_request
        from .config import find_repository_root
        director = Kantoku(self._config.repository_root or find_repository_root(), self._config, self._characters,
                           request.source_root if self._config.galgame.backend == "renpy" else None)
        def annotation_requests():
            for batch in batches:
                bound = bind_direction(build_annotation_request(
                    batch,
                    prompt_version=request.prompt_version,
                    annotation_config=self._config.annotation,
                    stage="cleaning",
                ), director)
                if self._config.cleaning.backend == "api":
                    yield bound
                else:
                    yield from split_directed_request(bound)

        batch_count = write_jsonl(annotation_requests(), layout.annotation_requests)
        source_audit = audit_dialogue(
            JsonlDialogueReader().read(layout.raw_dialogue),
            known_characters=set(self._characters),
        )
        layout.source_audit.write_text(
            json.dumps(source_audit, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        configuration_fingerprint = content_hash(
            {
                "batching": asdict(self._config.batching),
                "annotation": {
                    "allowed_emotions": sorted(
                        self._config.annotation.allowed_emotions
                    ),
                    "allowed_effects": sorted(
                        self._config.annotation.allowed_effects
                    ),
                    "maximum_length_ratio": (
                        self._config.annotation.maximum_length_ratio
                    ),
                    "minimum_length_ratio": (
                        self._config.annotation.minimum_length_ratio
                    ),
                },
                "audio": asdict(self._config.audio),
                "codex": asdict(self._config.codex),
                "galgame": {
                    "project": asdict(self._config.galgame),
                    "backend": dict(self._require_galgame_backend().configuration),
                },
            }
        )
        character_fingerprint = content_hash(
            {
                character_id: asdict(character)
                for character_id, character in sorted(self._characters.items())
            }
        )
        update_run_manifest(
            layout,
            {
                "dialogue_tab": {
                    "path": str(request.dialogue_tab_path.resolve()),
                    "sha256": dialogue_tab_hash,
                },
                "dialogue_count": dialogue_count,
                "batch_count": batch_count,
                "prompt_version": request.prompt_version,
                "dialogue_scope": (
                    request.dialogue_scope.name
                    if request.dialogue_scope is not None
                    else None
                ),
                "source_audit_valid": source_audit["valid"],
                "pipeline_configuration_hash": configuration_fingerprint,
                "character_configuration_hash": character_fingerprint,
            },
        )
        return dialogue_count, batch_count

    def validate(
        self,
        layout: ArtifactLayout,
        *,
        overrides_path: Path | None = None,
        retry: bool = False,
    ) -> ValidationSummary:
        """Validate untrusted annotations and apply trusted overrides."""

        result = self._validation.run(
            layout,
            overrides_path=overrides_path,
            retry=retry,
        )
        if not retry and result.retryable_count and layout.retry_responses.is_file():
            result = self._validation.run(layout, overrides_path=overrides_path, retry=True)
        return result

    def prepare_polish(self, layout: ArtifactLayout, *, prompt_path: Path | None = None) -> int:
        from .polish import PolishStage
        return PolishStage(self._config).prepare(layout, prompt_path=prompt_path)

    def validate_polish(self, layout: ArtifactLayout, *, retry: bool = False) -> ValidationSummary:
        from .polish import PolishStage
        return PolishStage(self._config).validate(layout, retry=retry)

    def plan_synthesis(self, layout: ArtifactLayout) -> tuple[int, int]:
        summary = self.validate_polish(layout)
        if summary.retryable_count or summary.review_required_count or summary.rejected_count or summary.batch_issue_count:
            raise ValueError("Polish is not fully accepted; synthesis cannot bypass polish validation")
        route_checker = getattr(self._synthesizer, "supports", None)
        direction_context = {key: value for request in read_jsonl(layout.annotation_requests)
                             for key, value in request.get("direction_context", {}).items()}
        planner = SynthesisPlanner(
            self._characters,
            audio_config=self._config.audio,
            synthesizer_configuration=(
                self._synthesizer.configuration
                if self._synthesizer is not None
                else None
            ),
            virtual_path_resolver=(
                self._require_galgame_backend().voice_virtual_path
            ),
            voice_route_available=(
                route_checker if callable(route_checker) else None
            ),
            direction_context=direction_context,
        )
        with JsonlIndex(layout.raw_dialogue, "id") as record_index:
            def records_and_results() -> Any:
                for value in read_jsonl(layout.polish.validated):
                    result = ValidatedAnnotation.from_dict(value)
                    raw_record = record_index.get(result.dialogue_id)
                    if raw_record is None:
                        raise ValueError(
                            "Validated result references unknown dialogue ID "
                            f"{result.dialogue_id!r}"
                        )
                    yield DialogueRecord.from_dict(raw_record), result

            with (
                AtomicJsonlWriter(layout.tts_jobs) as job_writer,
                AtomicJsonlWriter(layout.render_tasks) as render_writer,
                AtomicJsonlWriter(layout.synthesis_issues) as issue_writer,
                AtomicJsonlWriter(
                    layout.synthesis_adaptations
                ) as adaptation_writer,
            ):
                for item in planner.iter_plan(records_and_results()):
                    if isinstance(item, TtsJob):
                        job_writer.write(item.to_dict())
                        adapter = getattr(self._synthesizer, "adapt", None)
                        adaptation = (
                            adapter(item)
                            if callable(adapter)
                            else SpeechAdaptation(
                                job_id=item.id,
                                dialogue_id=item.dialogue_id,
                                backend=(
                                    self._synthesizer.name
                                    if self._synthesizer is not None
                                    else "unconfigured"
                                ),
                                text=item.text,
                                emotion=item.emotion,
                                parameters={
                                    "performance": item.performance.to_dict()
                                },
                            )
                        )
                        adaptation_writer.write(adaptation.to_dict())
                    elif isinstance(item, RenderTask):
                        render_writer.write(item.to_dict())
                    elif isinstance(item, SynthesisIssue):
                        issue_writer.write(item.to_dict())
                    else:
                        raise TypeError(f"Unexpected synthesis plan item: {item!r}")
        job_count = job_writer.count
        render_count = render_writer.count
        issue_count = issue_writer.count
        adaptation_count = adaptation_writer.count
        update_run_manifest(
            layout,
            {
                "tts_job_count": job_count,
                "render_task_count": render_count,
                "synthesis_issue_count": issue_count,
                "synthesis_adaptation_count": adaptation_count,
            },
        )
        return job_count, render_count

    def synthesize(self, layout: ArtifactLayout) -> tuple[int, int]:
        if self._synthesizer is None:
            raise RuntimeError("No speech synthesizer is configured")
        # Rebind jobs to accepted polish and current backend/profile settings.
        # Calling synthesize directly must not reuse a stale pre-polish plan.
        self.plan_synthesis(layout)
        update_run_manifest(layout, {"effects_configuration": self._config.effects.to_dict()})
        job_count = 0
        reused_job_count = 0
        jobs = (
            TtsJob.from_dict(value)
            for value in read_jsonl(layout.tts_jobs)
        )
        order_jobs = getattr(self._synthesizer, "order_jobs", None)
        scheduled_jobs = order_jobs(jobs) if callable(order_jobs) else jobs
        for job in scheduled_jobs:
            job_count += 1
            self._synthesizer.synthesize(job, layout.root)
            reused = getattr(self._synthesizer, "last_cache_hit", None)
            if reused is None:
                reused_job_count = None
            elif reused_job_count is not None:
                reused_job_count += int(reused)

        checker = AudioQualityChecker(self._config.audio)
        rendered = 0
        quality_failures = 0
        with (
            JsonlIndex(layout.tts_jobs, "id") as job_index,
            AtomicJsonlWriter(layout.audio_quality) as quality_writer,
            AtomicJsonlWriter(layout.audio_effects) as effects_writer,
        ):
            for value in read_jsonl(layout.render_tasks):
                task = RenderTask.from_dict(value)
                component_jobs: dict[str, TtsJob] = {}
                try:
                    for job_id in task.component_job_ids:
                        raw_job = job_index.get(job_id)
                        if raw_job is None:
                            raise KeyError(job_id)
                        component_jobs[job_id] = TtsJob.from_dict(raw_job)
                    path = self._renderer.render(task, component_jobs, layout.root)
                except (AudioRenderError, KeyError) as exc:
                    result = AudioQualityResult(
                        dialogue_id=task.dialogue_id,
                        path=str(layout.root / task.output_path),
                        valid=False,
                        duration_seconds=None,
                        issues=(str(exc),),
                    )
                else:
                    result = checker.check(task.dialogue_id, path)
                quality_writer.write(result.to_dict())
                if task.effects:
                    report = (layout.root / task.output_path).with_suffix(".effects.json")
                    if result.valid and report.is_file():
                        effects_writer.write(json.loads(report.read_text(encoding="utf-8")))
                    else:
                        effects_writer.write({"dialogue_id": task.dialogue_id, "requested": list(task.effects),
                                              "status": "failed" if not result.valid else "audit_unavailable",
                                              "issues": list(result.issues)})
                if result.valid:
                    rendered += 1
                else:
                    quality_failures += 1

        if not effects_writer.count:
            layout.audio_effects.unlink(missing_ok=True)

        integration_writer = VoiceManifestWriter()
        with (
            JsonlIndex(layout.raw_dialogue, "id") as record_index,
            JsonlIndex(layout.audio_quality, "dialogue_id") as quality_index,
        ):
            def manifest_entries() -> Any:
                for value in read_jsonl(layout.render_tasks):
                    task = RenderTask.from_dict(value)
                    raw_record = record_index.get(task.dialogue_id)
                    raw_quality = quality_index.get(task.dialogue_id)
                    if raw_record is None or raw_quality is None:
                        raise ValueError(
                            "Cannot build manifest for dialogue "
                            f"{task.dialogue_id!r}"
                        )
                    yield integration_writer.create_entry(
                        DialogueRecord.from_dict(raw_record),
                        task,
                        AudioQualityResult.from_dict(raw_quality),
                    )

            integration_writer.write_entries(
                layout.voice_manifest,
                manifest_entries(),
            )
        galgame_backend = self._require_galgame_backend()
        integration_artifact = (
            layout.galgame_artifacts / galgame_backend.integration_filename
        )
        with JsonlIndex(layout.audio_quality, "dialogue_id") as quality_index:
            galgame_backend.write_integration(
                integration_artifact,
                entries=(
                    (task.identifier, task.virtual_path)
                    for value in read_jsonl(layout.render_tasks)
                    for task in (RenderTask.from_dict(value),)
                    if (
                        (quality := quality_index.get(task.dialogue_id)) is not None
                        and quality.get("valid") is True
                    )
                ),
            )
        planning_failures = sum(1 for _ in read_jsonl(layout.synthesis_issues))
        incomplete = quality_failures or planning_failures
        update_run_manifest(
            layout,
            {
                "synthesis_status": ("partial" if rendered else "failed") if incomplete else "complete",
                "synthesizer": self._synthesizer.name,
                "tts_job_count": job_count,
                "tts_job_reused": reused_job_count,
                "rendered_count": rendered,
                "audio_quality_failure_count": quality_failures,
            },
        )
        if incomplete:
            raise AudioRenderError(
                f"Synthesis incomplete: {rendered} rendered, {quality_failures} audio failures, "
                f"{planning_failures} planning issues; see {layout.run_manifest}"
            )
        return job_count, rendered

    def build_release_bundle(self, layout: ArtifactLayout) -> int:
        """Create a game-relative directory without mutating the source release."""

        if layout.synthesis_issues.is_file() and any(read_jsonl(layout.synthesis_issues)):
            raise AudioRenderError("Cannot bundle an incomplete synthesis plan")
        with JsonlIndex(layout.audio_quality, "dialogue_id") as quality_index:
            # Validate every requested output before the installer writes a bundle.
            for value in read_jsonl(layout.render_tasks):
                quality = quality_index.get(value["dialogue_id"])
                if quality is None or quality.get("valid") is not True:
                    raise AudioRenderError(f"Cannot bundle failed or unchecked audio: {value['dialogue_id']}")

            def install_artifacts() -> Any:
                for value in read_jsonl(layout.render_tasks):
                    task = RenderTask.from_dict(value)
                    raw_quality = quality_index.get(task.dialogue_id)
                    if raw_quality is None:
                        raise ValueError(
                            "Cannot install dialogue without quality result: "
                            f"{task.dialogue_id!r}"
                        )
                    yield task, AudioQualityResult.from_dict(raw_quality)

            galgame_backend = self._require_galgame_backend()
            integration_artifact = (
                layout.galgame_artifacts / galgame_backend.integration_filename
            )
            result = galgame_backend.install_voice_bundle(
                layout.release_bundle,
                artifact_root=layout.root,
                integration_artifact=integration_artifact,
                voice_manifest=layout.voice_manifest,
                artifacts=install_artifacts(),
            )
        update_run_manifest(
            layout,
            {
                "release_bundle_audio_count": result.audio_count,
                "release_bundle_audio_archive": str(result.audio_archive),
                "release_patch": str(result.patch_path),
            },
        )
        return result.audio_count

    def run_from_responses(
        self,
        request: PipelineRequest,
        responses_path: Path,
        *,
        polish_responses_path: Path,
        cache_intermediates: bool = False,
    ) -> PipelineResult:
        """Run model-neutral stages from externally produced model responses."""

        layout = ArtifactLayout(request.artifact_root.resolve())
        source_dialogue = request.dialogue_tab_path.resolve()
        source_responses = responses_path.resolve()
        source_polish = polish_responses_path.resolve()
        inputs = [source_dialogue, source_responses, source_polish]
        if request.overrides_path is not None and request.overrides_path.is_file():
            inputs.append(request.overrides_path.resolve())
        for source in inputs:
            if not source.is_file():
                raise FileNotFoundError(source)
        files, directories = layout.reset_targets()
        # Only protect inputs that this cleanup can actually remove.
        with tempfile.TemporaryDirectory(prefix="lic-production-inputs-") as temporary:
            snapshots = {}
            for index, source in enumerate(inputs):
                if source in files or any(source.is_relative_to(directory) for directory in directories):
                    snapshot = Path(temporary) / f"{index}{source.suffix}"
                    shutil.copy2(source, snapshot)
                    snapshots[source] = snapshot
            try:
                layout.root.mkdir(parents=True, exist_ok=True)
                layout.reset_generated()
                if source_dialogue in snapshots:
                    source_dialogue.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snapshots[source_dialogue], source_dialogue)
                # Keep the original dialogue location: backends use its parent
                # for source context when no source_root was explicitly supplied.
                prepared_request = replace(request,
                    overrides_path=snapshots.get(request.overrides_path.resolve(), request.overrides_path) if request.overrides_path is not None else None)
                dialogue_count, batch_count = self.prepare(prepared_request)
                shutil.copy2(snapshots.get(source_responses, source_responses), layout.annotation_responses)
                self.validate(layout, overrides_path=prepared_request.overrides_path)
                self.prepare_polish(layout)
                shutil.copy2(snapshots.get(source_polish, source_polish), layout.polish.annotation_responses)
                validated = self.validate_polish(layout)
                tts_job_count, rendered_count = self.synthesize(layout)
                self.build_release_bundle(layout)
                result = PipelineResult(
                    artifacts=layout, dialogue_count=dialogue_count, batch_count=batch_count,
                    accepted_count=validated.accepted_count, review_count=validated.review_required_count,
                    retryable_count=validated.retryable_count, rejected_count=validated.rejected_count,
                    tts_job_count=tts_job_count, rendered_count=rendered_count,
                )
                update_run_manifest(layout, {"production": {
                    "annotation_responses": str(source_responses),
                    "polish_responses": str(source_polish),
                    "cache_intermediates": cache_intermediates,
                    "preserved_inputs": [str(path.relative_to(layout.root)) for path in inputs if path.is_relative_to(layout.root)],
                    "retained_paths": (["release_bundle", "lessons_in_cast_voice_patch.zip", "run_manifest.json"]
                                       if not cache_intermediates else ["all"]),
                }})
                if not cache_intermediates:
                    layout.discard_intermediates()
                return result
            finally:
                try:
                    for source, snapshot in snapshots.items():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snapshot, source)
                finally:
                    close = getattr(self._synthesizer, "close", None)
                    if close is not None:
                        close()
