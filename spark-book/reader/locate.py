"""Print the location of a cited file; execute nothing."""
import argparse,os
from pathlib import Path,PurePosixPath
p=argparse.ArgumentParser(description=__doc__);p.add_argument('path');a=p.parse_args()
rel=PurePosixPath(a.path)
if rel.is_absolute() or '..' in rel.parts or '\\' in a.path:p.error('supply a relative companion path')
roots=[('learner',Path(__file__).resolve().parents[1])]
for var,sub in [('SPARK_BOOK_EVIDENCE',''),('SPARK_BOOK_PUBLISHER','legacy')]:
 if os.environ.get(var):roots.append((var,Path(os.environ[var]).expanduser()/sub))
found=False
for label,root in roots:
 target=root/rel
 if target.is_file():print(str(target.resolve()));found=True
if not found:p.exit(2,'File not present in the selected roots. Read reader/EVIDENCE.txt.\n')
