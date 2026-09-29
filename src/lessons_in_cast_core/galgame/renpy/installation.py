"""Install rendered voices as an overlayable Ren'Py patch package."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from ...synthesis import AudioQualityResult, RenderTask
from ..api import GalgameInstallationResult
from .archive import RenPyArchiveWriter


class RenPyVoiceInstaller:
    """Create ``game/`` patch contents with audio packed into one RPA."""

    archive_filename = "lessons_in_cast_voice.rpa"
    patch_filename = "lessons_in_cast_voice_patch.zip"

    def __init__(self) -> None:
        self._archive_writer = RenPyArchiveWriter()

    def install(
        self,
        destination_root: Path,
        *,
        artifact_root: Path,
        voice_script: Path,
        voice_manifest: Path,
        artifacts: Iterable[tuple[RenderTask, AudioQualityResult]],
    ) -> GalgameInstallationResult:
        destination_root = destination_root.resolve()
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                dir=destination_root.parent,
                prefix=f".{destination_root.name}.install.",
            )
        )
        staging_game = staging_root / "game"
        staging_game.mkdir()
        archive_members: list[tuple[str, Path]] = []
        try:
            for task, quality in artifacts:
                if not quality.valid:
                    continue
                source = artifact_root / task.output_path
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Rendered audio is missing for {task.dialogue_id}: {source}"
                    )
                archive_members.append((task.virtual_path, source))

            archive_path = staging_game / self.archive_filename
            member_names = self._archive_writer.write(
                archive_path,
                archive_members,
            )
            shutil.copy2(voice_script, staging_game / voice_script.name)
            shutil.copy2(voice_manifest, staging_game / voice_manifest.name)
            package_manifest = {
                "format": "renpy-overlay-patch-v1",
                "audio_archive": f"game/{self.archive_filename}",
                "audio_count": len(member_names),
                "virtual_paths": list(member_names),
            }
            (staging_game / "lessons_in_cast_patch.json").write_text(
                json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            if destination_root.exists():
                shutil.rmtree(destination_root)
            staging_root.replace(destination_root)
            game_root = destination_root / "game"
            final_archive = game_root / self.archive_filename
            patch_path = destination_root.parent / self.patch_filename
            temporary_patch = patch_path.with_suffix(".zip.tmp")
            temporary_patch.unlink(missing_ok=True)
            with zipfile.ZipFile(
                temporary_patch,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as package:
                for path in sorted(game_root.rglob("*")):
                    if path.is_file():
                        package.write(
                            path,
                            path.relative_to(destination_root).as_posix(),
                        )
            temporary_patch.replace(patch_path)
            return GalgameInstallationResult(
                content_root=game_root,
                audio_count=len(member_names),
                audio_archive=final_archive,
                patch_path=patch_path,
            )
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
