"""Write Ren'Py's callable auto-voice resolver."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


class RenPyVoiceScriptWriter:
    """Write the Ren'Py script that maps dialogue IDs to virtual audio paths."""

    def write(
        self,
        destination: Path,
        *,
        entries: Iterable[tuple[str, str]],
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write("init -100 python:\n")
                output.write("    _lessons_in_cast_voice_paths = {\n")
                for identifier, virtual_path in entries:
                    output.write(
                        f"        {json.dumps(identifier)}: "
                        f"{json.dumps(virtual_path)},\n"
                    )
                output.write("    }\n\n")
                output.write("    def _lessons_in_cast_auto_voice(identifier):\n")
                output.write(
                    "        return _lessons_in_cast_voice_paths.get(identifier)\n\n"
                )
                output.write("    config.auto_voice = _lessons_in_cast_auto_voice\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination
