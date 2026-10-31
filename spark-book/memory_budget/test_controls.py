"""CPU editorial checks only. Never allocates CUDA memory."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('memory_probe', ROOT/'probe.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Controls(unittest.TestCase):
    def test_guard_requires_guest_headroom(self):
        state = {'guest': {'MemAvailable_bytes': m.MIN_GUEST_AVAILABLE-1},
                 'cuda_free_bytes': 30*m.GIB}
        with self.assertRaisesRegex(RuntimeError, 'guest'):
            m.require_headroom(state, m.GIB)

    def test_guard_preserves_cuda_reserve(self):
        state = {'guest': {'MemAvailable_bytes': 3*m.GIB},
                 'cuda_free_bytes': m.GIB+m.MIN_CUDA_RESERVE-1}
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            m.require_headroom(state, m.GIB)
        state['cuda_free_bytes'] += 1
        m.require_headroom(state, m.GIB)

    def test_wrapper_rejects_oversized_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'never-created'
            p = subprocess.run([sys.executable, str(ROOT/'run.py'), '--out', str(out),
                                '--allocate-gib', '19', '--host-note', 'CPU test'],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(out.exists())

    def test_wrapper_requires_host_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'never-created'
            p = subprocess.run([sys.executable, str(ROOT/'run.py'), '--out', str(out),
                                '--allocate-gib', '1'], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(out.exists())

    def test_wrapper_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp)/'sentinel.txt'; sentinel.write_text('retain')
            p = subprocess.run([sys.executable, str(ROOT/'run.py'), '--out', tmp],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'retain')
            self.assertFalse((Path(tmp)/'metadata.json').exists())

    def test_full_tile_and_tail_readback_cpu(self):
        import numpy as np
        class Tensor:
            def __init__(self, array): self.array = array
            def numel(self): return self.array.size
            def __getitem__(self, part): return Tensor(self.array[part])
            def __eq__(self, value): return self.array == value
        class CPUFacade:
            all = staticmethod(np.all)
            class cuda:
                synchronize = staticmethod(lambda device: None)
        blocks = [Tensor(np.full(m.CHECK_TILE+3, value, dtype=np.uint8))
                  for value in (1, 2)]
        self.assertEqual(m.verify_blocks(CPUFacade, blocks), 2*(m.CHECK_TILE+3))
        blocks[0].array[-1] = 9
        with self.assertRaisesRegex(RuntimeError, 'CORRUPTION'):
            m.verify_blocks(CPUFacade, blocks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
