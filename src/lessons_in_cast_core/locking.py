"""Cross-platform, non-blocking exclusive file locks."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path``; raise BlockingIOError if it is held.

    The lock is advisory, released when the block exits or the process dies,
    and the lock file itself is left in place.
    """

    with open(path, "a+b") as stream:
        if os.name == "nt":
            # Windows locks byte ranges; one byte at offset 0 is enough, even
            # past the end of an empty file.
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError(f"{path} is locked by another process") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
