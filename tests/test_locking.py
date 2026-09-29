from __future__ import annotations

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
