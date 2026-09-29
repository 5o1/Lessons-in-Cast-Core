"""Filesystem location of the installed package."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Directory to put on PYTHONPATH so subprocesses can import this package.
SOURCE_ROOT = PACKAGE_ROOT.parent


def venv_python(environment: Path) -> Path:
    """Return the interpreter inside a virtual environment on this platform."""

    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"
