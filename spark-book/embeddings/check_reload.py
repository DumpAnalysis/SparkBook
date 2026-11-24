"""Compare the selected pre-save vectors to a single independent checkpoint reload."""
from pathlib import Path
import sys
import tempfile
import numpy as np

PROBES = ['GPU memory allocation failed.', 'Embedding vectors require validation.']
ATOL = 1e-6
RTOL = 1e-5


def probe_vectors(model):
    model.eval()
    vectors = np.asarray(model.encode(PROBES, normalize_embeddings=True, show_progress_bar=False)).copy()
    if vectors.ndim != 2 or vectors.shape[0] != len(PROBES) or not np.isfinite(vectors).all():
        raise ValueError('Selected encoder returned invalid reference vectors')
    return vectors


def compare(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Probe vector shape or finiteness mismatch')
    error = float(np.max(np.abs(expected-actual)))
    if not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        raise ValueError(f'Checkpoint probe mismatch: max_abs={error:.8g}, atol={ATOL}, rtol={RTOL}')
    return error


def checked_save(model, output):
    reference=probe_vectors(model)
    model.save_pretrained(str(output))
    np.save(Path(output)/'selected-probe.npy', reference, allow_pickle=False)


def self_test():
    reference=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    class FixtureEncoder:
        def __init__(self, vectors):
            self.vectors = vectors.copy()

        def eval(self):
            pass

        def encode(self, texts, **options):
            return self.vectors[:len(texts)]

        def save_pretrained(self, directory):
            np.save(Path(directory)/'fixture.npy', self.vectors, allow_pickle=False)

    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        selected = FixtureEncoder(reference)
        checked_save(selected, directory)
        selected.vectors[:] = 0
        reloaded = FixtureEncoder(np.load(directory/'fixture.npy', allow_pickle=False))
        before = np.load(directory/'selected-probe.npy', allow_pickle=False)
        print('Reference round trip', compare(before, probe_vectors(reloaded)))
    for name,other in [('wrong checkpoint',reference[::-1]),('nonfinite',reference*np.nan),('wrong shape',reference[:1])]:
        try:compare(reference, other)
        except ValueError: print(name,'rejected')
        else: raise AssertionError(name+' escaped detection')


if __name__=='__main__':
    if sys.argv[1:]!=['--self-test']: raise SystemExit('Usage: python embeddings/check_reload.py --self-test')
    self_test()
