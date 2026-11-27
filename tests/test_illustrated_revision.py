"""Reader regressions for an active native binary and incomplete GPU activity."""
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest

BOOK = Path(__file__).resolve().parents[1] / 'spark-book'
sys.path.insert(0, str(BOOK))
from setup_wsl import install_runtime
from cuda import trace_sqlite


class IllustratedRevision(unittest.TestCase):
    def test_install_preserves_running_binary_and_repairs_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'native.tar'
            with tarfile.open(archive, 'w') as tar:
                tar.add(Path(sys.executable).resolve(), 'release/server')
                alias = tarfile.TarInfo('release/current')
                alias.type, alias.linkname = tarfile.SYMTYPE, 'server'
                tar.addfile(alias)
            dest, state = root / 'linux-bin', root / 'state.json'
            install_runtime([archive], dest, state)
            exe = dest / 'release/server'
            initial = exe.stat()
            state.touch()
            with subprocess.Popen([dest / 'release/current', '-c', 'import time; time.sleep(30)']) as child:
                try:
                    install_runtime([archive], dest, state)
                    self.assertEqual((exe.stat().st_ino, exe.stat().st_mtime_ns), (initial.st_ino, initial.st_mtime_ns))
                    self.assertIsNone(child.poll())
                    (dest / 'release/current').unlink()
                    with self.assertRaisesRegex(SystemExit, 'Stop the recorded server'):
                        install_runtime([archive], dest, state)
                finally:
                    child.terminate()
            state.unlink()
            exe.write_bytes(b'corrupt')
            install_runtime([archive], dest, state)
            self.assertEqual(exe.read_bytes(), Path(sys.executable).read_bytes())
            self.assertEqual((dest / 'release/current').readlink(), Path('server'))

    def test_trace_excludes_incomplete_intervals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.sqlite'
            trace_sqlite.fixture(path)
            with sqlite3.connect(path) as con:
                con.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,999,1),(100,100,1)')
            self.assertEqual(trace_sqlite.summarize(path)[1], [('transform', 2, 500)])
            with sqlite3.connect(path) as con:
                con.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>start AND start>0')
            with self.assertRaisesRegex(ValueError, 'No positive'):
                trace_sqlite.summarize(path)
