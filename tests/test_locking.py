from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lessons_in_cast_core.locking import exclusive_lock


class ExclusiveLockTests(unittest.TestCase):
    def test_second_holder_is_rejected_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".lock"
            with exclusive_lock(path):
                with self.assertRaises(BlockingIOError):
                    with exclusive_lock(path):
                        pass
            with exclusive_lock(path):
                pass
            self.assertTrue(path.is_file())

    def test_process_exit_releases_lock_without_deleting_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".lock"
            script = (
                "import os, sys; from pathlib import Path; "
                "from lessons_in_cast_core.locking import exclusive_lock\n"
                "with exclusive_lock(Path(sys.argv[1])): os._exit(23)\n"
            )
            result = subprocess.run([sys.executable, "-c", script, str(path)],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue(path.exists())
            with exclusive_lock(path):
                pass
