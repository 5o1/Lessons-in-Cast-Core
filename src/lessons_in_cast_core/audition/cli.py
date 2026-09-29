"""Collect audition contexts, validate sides, render takes, and record callbacks."""

import argparse
import json
from pathlib import Path
import sys

from ..config import find_repository_root, load_dialogue_sources, load_pipeline_config, load_workspace_config
from ..galgame import DialogueExtractionRequest, load_galgame_backend
from .collection import collect_contexts
from .rendering import render_project, save
from .types import load_project, slug
from .cleaning import prepare_cleaning, workflow
from ..pipeline import DialoguePipeline
from ..characters import load_characters
from ..annotation.codex import CodexWorkspace
from ..workflow.artifacts import ArtifactLayout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lessons-in-cast-audition")
    parser.add_argument("--root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Gather all-export contexts for manual/Codex side selection")
    collect.add_argument("--character", required=True)
    collect.add_argument("--dialogue", type=Path, help="Existing full-game export; otherwise invoke the game backend")
    collect.add_argument("--speaker", action="append", default=[], help="Additional ambiguous speaker to include for review, not automatic attribution")
    collect.add_argument("--collection", required=True, help="New build/auditions collection directory name")
    validate = commands.add_parser("validate")
    validate.add_argument("project", type=Path)
    prepare = commands.add_parser("prepare", help="Prepare source/context and director notes for normal Codex cleaning")
    prepare.add_argument("project", type=Path)
    prepare.add_argument("--dialogue", required=True, type=Path)
    prepare.add_argument("--run", required=True)
    for name in ("cleaning-next", "cleaning-import", "polish-prepare", "polish-next", "polish-import", "polish-validate"):
        command = commands.add_parser(name)
        command.add_argument("--cleaning", required=True, type=Path)
    render = commands.add_parser("render")
    render.add_argument("project", type=Path)
    render.add_argument("--run", required=True)
    render.add_argument("--cleaning", required=True, type=Path)
    render.add_argument("--candidate", default="default")
    render.add_argument("--profile", type=Path)
    render.add_argument("--reference-audio", type=Path)
    render.add_argument("--case", action="append", default=[])
    review = commands.add_parser("review", help="Record a listening decision without changing profiles")
    review.add_argument("--run", required=True)
    review.add_argument("--case", required=True)
    review.add_argument("--decision", required=True, choices=["undecided", "shortlist", "callback", "reject"])
    review.add_argument("--notes", required=True)
    args = parser.parse_args(argv)
    root = (args.root or find_repository_root()).resolve()
    rooted = lambda path: path if path.is_absolute() else root / path
    try:
        if args.command == "validate":
            project = load_project(rooted(args.project))
            print(json.dumps({"project": project.id, "cases": len(project.cases), "coverage": project.coverage}))
        elif args.command == "prepare":
            slug(args.run)
            print(json.dumps(prepare_cleaning(root, rooted(args.project), rooted(args.dialogue), root / "build/auditions" / args.run)))
        elif args.command in {"cleaning-next", "cleaning-import", "polish-prepare", "polish-next", "polish-import", "polish-validate"}:
            directory = rooted(args.cleaning)
            layout = ArtifactLayout(directory)
            stage = "polish" if args.command.startswith("polish-") else "cleaning"
            config = load_pipeline_config(repository_root=root)
            pipeline = DialoguePipeline(config=config, characters=load_characters(repository_root=root))
            if args.command == "polish-prepare":
                pipeline.validate(layout)
                pipeline.prepare_polish(layout, prompt_path=root / config.codex.polish_prompt_path)
            if args.command == "polish-validate":
                print(json.dumps(pipeline.validate_polish(layout).to_dict()))
                return 0
            if stage == "polish":
                from ..polish import PolishStage
                PolishStage(config).check_inputs(layout)
                layout = layout.polish
            if getattr(config, stage).backend == "api" and (args.command.endswith("next") or args.command == "polish-prepare"):
                if args.command == "polish-prepare":
                    print(json.dumps({"status": "awaiting_api_annotation"}))
                else:
                    from ..annotation.agent import AnnotationAgent
                    print(json.dumps(AnnotationAgent(root, config, load_characters(repository_root=root), stage).run(layout)))
                return 0
            exchange = workflow(root, directory, stage=stage)
            workspace = CodexWorkspace(layout.root / "codex")
            if args.command.endswith("next") or args.command == "polish-prepare":
                print(json.dumps(exchange.export_next(layout.annotation_requests, layout.annotation_responses, workspace, batches_per_packet=1).to_dict()))
            else:
                print(json.dumps({"imported": exchange.import_outbox(layout.annotation_requests, layout.annotation_responses, workspace)}))
        elif args.command == "render":
            output = render_project(root, rooted(args.project), args.run, profile=args.profile,
                cleaning=rooted(args.cleaning),
                reference_audio=rooted(args.reference_audio) if args.reference_audio else None,
                case_ids=tuple(args.case), candidate=args.candidate)
            print(json.dumps({"output": str(output)}))
        elif args.command == "collect":
            slug(args.collection)
            output = root / "build/auditions" / args.collection
            dialogue = rooted(args.dialogue) if args.dialogue else output.parent / f"{args.collection}.tab"
            if not args.dialogue:
                if dialogue.exists():
                    raise FileExistsError(f"Export exists; pass --dialogue explicitly: {dialogue}")
                config = load_pipeline_config(repository_root=root)
                backend = load_galgame_backend(config.galgame.backend)
                workspace = load_workspace_config(repository_root=root)
                dialogue.parent.mkdir(parents=True, exist_ok=True)
                backend.extract_dialogue(DialogueExtractionRequest(workspace.release_path, dialogue,
                    source_paths=load_dialogue_sources(repository_root=root)))
            report = collect_contexts(root, dialogue, args.character, output, tuple(args.speaker))
            print(json.dumps(report))
        else:
            slug(args.run)
            path = root / "build/auditions" / args.run / "casting_notes.json"
            notes = json.loads(path.read_text(encoding="utf-8"))
            if args.case not in notes:
                raise ValueError(f"Unknown take: {args.case}")
            notes[args.case] = {"decision": args.decision, "notes": args.notes}
            save(path, notes)
            print(json.dumps({"notes": str(path)}))
    except (OSError, ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as exc:
        print(f"Audition failed: {exc}", file=sys.stderr)
        return 1
    return 0
