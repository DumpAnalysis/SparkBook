"""Print the path of a file in the learner companion."""
import argparse
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('path')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
rel = Path(a.path)
if rel.is_absolute() or '..' in rel.parts:
    p.error('supply a relative companion path')
target = (root / rel).resolve()
if not target.is_relative_to(root) or not target.is_file():
    p.exit(2, 'File not found in the companion. Use reader/guide.py for the portable procedure.\n')
print(target)
