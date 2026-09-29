"""Command-line entry point for reproducible pipeline stages."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Sequence

from .characters import CharacterDefinition, load_characters
from .config import (
    PipelineConfig,
    find_repository_root,
    load_dialogue_sources,
    load_pipeline_config,
    load_workspace_config,
)
from .dialogue import load_dialogue_scopes
from .galgame import DialogueExtractionRequest, load_galgame_backend
from .model_registry import load_model_registry
from .pipeline import ArtifactLayout, DialoguePipeline, PipelineRequest
from .workspace_init import DEFAULT_RELEASE, initialize_workspace
from .synthesis.profiles.loader import NoVoiceProfilesError
from .synthesis import (
    ReferenceBuildRequest,
    ReferenceVoicePipeline,
    VoiceProfileSynthesizer,
    load_configured_voice_profiles,
    load_voice_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lessons-in-cast")
    parser.add_argument(
        "--root",
        type=Path,
        help="Repository root; discovered automatically by default.",
    )
    parser.add_argument(
        "--build-dir",
        dest="build_dir",
        type=Path,
        default=Path("build/current"),
        help="Active run directory relative to the repository root.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init",
        help="Create a workspace with template configuration in the current directory.",
    )
    init.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="Game release directory to record in configs/workspace.toml.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files with the templates.",
    )
    from .emotion_presets import register_cli
    register_cli(commands)
    commands.add_parser("check-config", help="Validate all project configuration.")
    voices = commands.add_parser("voice-tags", help="List dynamic voice tags without preparing or loading a model.")
    voices.add_argument("--character", required=True)
    voices.add_argument("--profile", type=Path, help="Profile entrypoint; defaults to the character's global profile")
    voices.add_argument("--reference-audio", type=Path, help="Temporarily query an overridden default reference directory")
    extract = commands.add_parser(
        "extract",
        help="Generate the dialogue export with the configured galgame backend.",
    )
    extract.add_argument("--executable", type=Path)
    extract.add_argument("--language")
    prepare = commands.add_parser(
        "prepare",
        help="Convert dialogue.tab to raw JSONL and annotation requests.",
    )
    prepare.add_argument("--input", type=Path)
    prepare.add_argument("--scope", help="Named dialogue scope from configs/demo_scopes.toml.")
    codex_next = commands.add_parser(
        "codex-next",
        help="Export the next continuous transcript packet for Codex.",
    )
    codex_next.add_argument("--retry", action="store_true")
    codex_next.add_argument("--batches-per-packet", type=int)
    codex_next.add_argument("--stage", choices=["cleaning", "polish"], default="cleaning")
    codex_import = commands.add_parser(
        "codex-import",
        help="Import and split the active Codex packet response.",
    )
    codex_import.add_argument("--retry", action="store_true")
    codex_import.add_argument("--replace", action="store_true")
    codex_import.add_argument("--stage", choices=["cleaning", "polish"], default="cleaning")
    codex_status = commands.add_parser(
        "codex-status",
        help="Report progress for the Codex annotation pass.",
    )
    codex_status.add_argument("--retry", action="store_true")
    codex_status.add_argument("--stage", choices=["cleaning", "polish"], default="cleaning")
    annotate = commands.add_parser("annotate", help="Execute the configured annotation backend, or export its Codex task")
    annotate.add_argument("--stage", choices=["cleaning", "polish"], default="cleaning")
    annotate.add_argument("--retry", action="store_true")
    annotate.add_argument("--preview", action="store_true", help="Write API messages without making a request")
    commands.add_parser("polish-prepare", help="Prepare a separate acting pass from accepted cleaning results")
    polish_validate = commands.add_parser("polish-validate")
    polish_validate.add_argument("--retry", action="store_true")
    validate = commands.add_parser(
        "validate",
        help="Validate annotation responses and apply overrides.",
    )
    validate.add_argument("--retry", action="store_true")
    commands.add_parser("plan-tts", help="Create cached TTS and render jobs.")
    commands.add_parser(
        "prepare-voices",
        help="Prepare generated dependencies for configured voice profiles.",
    )
    commands.add_parser(
        "synthesize", help="Generate audio with configured voice profiles."
    )
    build_reference = commands.add_parser(
        "build-reference",
        help="Build a character reference WAV from a sample-audio directory.",
    )
    build_reference.add_argument("--character", required=True)
    build_reference.add_argument("--input-dir", type=Path, required=True)
    build_reference.add_argument("--output", type=Path)
    commands.add_parser("bundle", help="Build a game-relative voice release bundle.")
    production = commands.add_parser(
        "run-production",
        help=(
            "Validate external annotations, synthesize, and build the "
            "galgame release bundle."
        ),
    )
    production.add_argument("--input", type=Path, required=True)
    production.add_argument("--responses", type=Path, required=True)
    production.add_argument("--polish-responses", type=Path, required=True)
    production.add_argument("--scope", help="Named dialogue scope from configs/demo_scopes.toml.")
    production.add_argument(
        "--cache-intermediates",
        action="store_true",
        help="Retain intermediate JSONL and audio after a successful run.",
    )
    return parser


def _voice_profile_synthesizer(
    root: Path,
    config: PipelineConfig,
    characters: dict[str, CharacterDefinition],
) -> VoiceProfileSynthesizer:
    return load_configured_voice_profiles(root, config, characters)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result = initialize_workspace(
            args.root or Path.cwd(), release=args.release, force=args.force
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    root = (args.root or find_repository_root()).resolve()
    if args.command == "emotion-presets":
        from .emotion_presets import cli_main
        return cli_main(args, root)
    artifact_root = args.build_dir
    if not artifact_root.is_absolute():
        artifact_root = root / artifact_root
    layout = ArtifactLayout(artifact_root.resolve())
    config = load_pipeline_config(repository_root=root)
    workspace = load_workspace_config(repository_root=root)
    galgame_backend = load_galgame_backend(config.galgame.backend)
    characters = load_characters(repository_root=root)
    sources = load_dialogue_sources(repository_root=root)
    scopes = load_dialogue_scopes(repository_root=root)
    scope_name = getattr(args, "scope", None)
    dialogue_scope = scopes.get(scope_name) if scope_name else None
    if scope_name and dialogue_scope is None:
        raise ValueError(f"Unknown dialogue scope: {scope_name!r}")

    if args.command == "check-config":
        try:
            voice_profile_synthesizer = load_configured_voice_profiles(
                root, config, characters
            )
        except NoVoiceProfilesError:
            # A new workspace is valid before any character is voiced.
            voice_profile_ids = []
        else:
            try:
                voice_profile_ids = sorted(
                    voice_profile_synthesizer.configuration["profiles"]
                )
            finally:
                voice_profile_synthesizer.close()
        print(
            json.dumps(
                {
                    "release_path": str(workspace.release_path),
                    "dialogue_sources": len(sources),
                    "dialogue_scopes": sorted(scopes),
                    "characters": len(characters),
                    "voice_profiles": voice_profile_ids,
                    "galgame_backend": galgame_backend.backend_id,
                    "codex_batches_per_packet": config.codex.batches_per_packet,
                    "codex_source_files": list(config.codex.source_files),
                    "codex_prompt_path": config.codex.prompt_path,
                    "codex_prompt_version": config.codex.prompt_version,
                    "cleaning_backend": config.cleaning.backend,
                    "polish_backend": config.polish.backend,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "prepare-voices":
        voice_profile_synthesizer = _voice_profile_synthesizer(root, config, characters)
        try:
            prepared = voice_profile_synthesizer.prepare()
        finally:
            voice_profile_synthesizer.close()
        print(
            json.dumps(
                {key: [str(path) for path in paths] for key, paths in prepared.items()},
                indent=2,
            )
        )
        return 0

    if args.command == "voice-tags":
        character = characters[args.character]
        entrypoint = args.profile or Path(character.default_voice_profile)
        if args.profile is None and not character.default_voice_profile:
            raise ValueError("No default profile; pass --profile")
        profile = load_voice_profile(root, entrypoint, character, config,
                                     model_registry=load_model_registry(repository_root=root))
        try:
            if args.reference_audio is not None:
                reference = args.reference_audio
                profile.override_reference_audio(reference if reference.is_absolute() else root / reference)
            try:
                result = {"supported": True, "tags": list(profile.list_voice_tags())}
            except NotImplementedError as exc:
                result = {"supported": False, "tags": [], "reason": str(exc)}
            print(json.dumps({"profile": str(entrypoint), **result}, indent=2))
        finally:
            profile.close()
        return 0

    if args.command == "build-reference":
        character = characters.get(args.character)
        if character is None:
            raise ValueError(f"Unknown character ID: {args.character!r}")
        if not character.default_voice_profile:
            raise ValueError(
                f"Character {args.character!r} has no default_voice_profile"
            )
        voice_profile = load_voice_profile(
            root,
            Path(character.default_voice_profile),
            character,
            config,
            model_registry=load_model_registry(repository_root=root),
        )
        if not isinstance(voice_profile, ReferenceVoicePipeline):
            voice_profile.close()
            raise TypeError(
                f"Voice profile for {character.id!r} does not build references"
            )
        default_request = voice_profile.default_reference_request
        input_directory = (
            args.input_dir
            if args.input_dir.is_absolute()
            else root / args.input_dir
        )
        output_path = args.output or default_request.output_path
        if not output_path.is_absolute():
            output_path = root / output_path
        try:
            result = voice_profile.build_reference(
                ReferenceBuildRequest(input_directory, output_path)
            )
        finally:
            voice_profile.close()
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "extract":
        executable = args.executable
        if executable is not None and not executable.is_absolute():
            executable = root / executable
        request = DialogueExtractionRequest(
            release_path=workspace.release_path,
            output_path=layout.dialogue_tab,
            source_paths=sources,
            language=args.language,
            executable_path=executable,
        )
        pipeline = DialoguePipeline(
            config=config,
            characters=characters,
            galgame_backend=galgame_backend,
        )
        path = pipeline.extract(request)
        print(path)
        return 0

    if args.command in {"annotate", "codex-next", "codex-import", "codex-status"}:
        from .annotation import CodexAnnotationWorkflow, CodexWorkspace

        base_layout = layout
        if args.stage == "polish":
            from .polish import PolishStage
            PolishStage(config).check_inputs(base_layout)
            layout = layout.polish
        if args.command == "annotate" and getattr(config, args.stage).backend == "api":
            from .annotation.agent import AnnotationAgent
            print(json.dumps(AnnotationAgent(root, config, characters, args.stage).run(
                layout, retry=args.retry, preview=args.preview, source_files=config.codex.source_files)))
            return 0
        retry = args.retry
        requests_path = (
            layout.retry_requests if retry else layout.annotation_requests
        )
        responses_path = (
            layout.retry_responses if retry else layout.annotation_responses
        )
        workspace = CodexWorkspace(
            layout.root / "codex" / ("retry" if retry else "initial")
        )
        prefix = f"python -m lessons_in_cast_core --root {shlex.quote(str(root))} --build-dir {shlex.quote(str(base_layout.root))}"
        options = f" --stage {args.stage}" + (" --retry" if retry else "")
        workflow = CodexAnnotationWorkflow(
            root / (config.codex.polish_prompt_path if args.stage == "polish" else config.codex.prompt_path),
            {
                character_id: character.name
                for character_id, character in characters.items()
            },
            config.codex.source_files,
            task_commands=(f"{prefix} codex-import{options}", f"{prefix} codex-next{options}"),
        )
        if args.command in {"annotate", "codex-next"}:
            packet = workflow.export_next(
                requests_path,
                responses_path,
                workspace,
                batches_per_packet=(
                    args.batches_per_packet
                    if getattr(args, "batches_per_packet", None) is not None
                    else config.codex.batches_per_packet
                ),
            )
            print(json.dumps(packet.to_dict()))
        elif args.command == "codex-import":
            imported = workflow.import_outbox(
                requests_path,
                responses_path,
                workspace,
                replace=args.replace,
            )
            print(json.dumps({"imported_batches": imported}))
        else:
            print(json.dumps(workflow.status(requests_path, responses_path).to_dict()))
        return 0

    synthesizer = None
    if args.command in {"plan-tts", "synthesize", "run-production"}:
        synthesizer = _voice_profile_synthesizer(root, config, characters)
    pipeline = DialoguePipeline(
        config=config,
        characters=characters,
        synthesizer=synthesizer,
        galgame_backend=galgame_backend,
    )
    if args.command == "prepare":
        source = args.input or layout.dialogue_tab
        counts = pipeline.prepare(
            PipelineRequest(
                artifact_root=layout.root,
                dialogue_tab_path=source,
                allowed_sources=sources,
                dialogue_scope=dialogue_scope,
                source_root=workspace.release_path,
                prompt_version=config.codex.prompt_version,
            )
        )
        print(json.dumps({"dialogue_count": counts[0], "batch_count": counts[1]}))
    elif args.command == "validate":
        summary = pipeline.validate(
            layout,
            overrides_path=root / "configs" / "overrides.toml",
            retry=args.retry,
        )
        print(json.dumps(summary.to_dict()))
    elif args.command == "plan-tts":
        jobs, renders = pipeline.plan_synthesis(layout)
        print(json.dumps({"tts_job_count": jobs, "render_task_count": renders}))
    elif args.command == "polish-prepare":
        print(json.dumps({"batches": pipeline.prepare_polish(layout, prompt_path=root / config.codex.polish_prompt_path)}))
    elif args.command == "polish-validate":
        print(json.dumps(pipeline.validate_polish(layout, retry=args.retry).to_dict()))
    elif args.command == "synthesize":
        jobs, rendered = pipeline.synthesize(layout)
        print(json.dumps({"tts_job_count": jobs, "rendered_count": rendered}))
    elif args.command == "bundle":
        print(
            json.dumps(
                {"bundled_audio_count": pipeline.build_release_bundle(layout)}
            )
        )
    elif args.command == "run-production":
        input_path = args.input if args.input.is_absolute() else root / args.input
        responses_path = (
            args.responses if args.responses.is_absolute() else root / args.responses
        )
        result = pipeline.run_from_responses(
            PipelineRequest(
                artifact_root=layout.root,
                dialogue_tab_path=input_path,
                allowed_sources=sources,
                overrides_path=root / "configs" / "overrides.toml",
                dialogue_scope=dialogue_scope,
                source_root=workspace.release_path,
                prompt_version=config.codex.prompt_version,
            ),
            responses_path,
            polish_responses_path=(args.polish_responses if args.polish_responses.is_absolute() else root / args.polish_responses),
            cache_intermediates=args.cache_intermediates,
        )
        print(
            json.dumps(
                {
                    "dialogue_count": result.dialogue_count,
                    "batch_count": result.batch_count,
                    "accepted_count": result.accepted_count,
                    "review_count": result.review_count,
                    "retryable_count": result.retryable_count,
                    "rejected_count": result.rejected_count,
                    "tts_job_count": result.tts_job_count,
                    "rendered_count": result.rendered_count,
                    "release_bundle": str(result.artifacts.release_bundle),
                    "release_patch": str(result.artifacts.release_patch),
                    "intermediates_cached": args.cache_intermediates,
                },
                indent=2,
            )
        )
    return 0
