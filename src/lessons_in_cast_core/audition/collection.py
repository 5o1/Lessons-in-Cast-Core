"""Collect whole dialogue contexts for a separate semantic casting session."""

from collections import defaultdict
import json
from pathlib import Path
import re

from ..characters import load_characters
from ..config import load_pipeline_config, load_workspace_config
from ..galgame import load_galgame_backend
from ..hashing import file_hash


def collect_contexts(root: Path, dialogue: Path, character: str, output: Path, speakers: tuple[str, ...] = ()) -> dict:
    """No chapter filter or automatic emotion-to-state clustering is applied.

    Mention matches and aliases are candidates, not verified speaker identity.
    Context boundaries come from the game backend, not an inferred scene graph.
    """
    if output.exists():
        raise FileExistsError(f"Choose a new collection directory: {output}")
    definition = load_characters(repository_root=root)[character]
    project = load_pipeline_config(repository_root=root)
    workspace = load_workspace_config(repository_root=root)
    backend = load_galgame_backend(project.galgame.backend)
    groups = defaultdict(list)
    matched = set()
    sources = set()
    count = 0
    mentions = re.compile(r"(?<!\w)" + re.escape(definition.name) + r"(?!\w)", re.I)
    for row in backend.read_dialogue(dialogue, source_root=workspace.release_path):
        key = (row.filename, row.label)
        groups[key].append(row)
        sources.add(row.filename)
        count += 1
        if row.character in {character, *speakers} or mentions.search(row.dialogue):
            matched.add(key)
    output.mkdir(parents=True)
    with (output / "contexts.jsonl").open("w", encoding="utf-8") as target:
        for filename, label in sorted(matched):
            rows = sorted(groups[(filename, label)], key=lambda row: (row.line_number, row.sequence))
            target.write(json.dumps({"filename": filename, "label": label,
                "selection_status": "unreviewed_candidate",
                "dialogue": [row.to_dict() for row in rows]}, ensure_ascii=False) + "\n")
    report = {"character": character, "aliases_for_review": list(speakers), "dialogue_export": str(dialogue),
              "dialogue_sha256": file_hash(dialogue), "release": str(workspace.release_path),
              "rows_read": count, "files_in_export": sorted(sources), "candidate_contexts": len(matched),
              "chapter_filter": None, "semantic_review_complete": False,
              "limitations": "Coverage is limited to the supplied export. Mentions, anonymous speakers, routes and impersonation require semantic review; no automatic completeness claim."}
    (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
