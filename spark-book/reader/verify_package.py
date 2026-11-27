"""Check the pristine learner package and optionally its matching book."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--book', type=Path, action='append', default=[], help='verify a matching DOCX or PDF; repeat to verify both')
args = parser.parse_args()
name = 'VERSION59_MANIFEST.json'
manifest = json.loads((root / name).read_text())


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


expected = {name}
for row in manifest['files']:
    path = root / row['path']
    if not path.is_file() or path.is_symlink() or checksum(path) != row['sha256']:
        parser.exit(1, 'Missing, altered or invalid package file: ' + row['path'] + '\n')
    expected.add(row['path'])
found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
if found != expected:
    extras = sorted(found - expected)
    missing = sorted(expected - found)
    detail = []
    for label, paths in [('unexpected', extras), ('missing', missing)]:
        if paths:
            detail.append(f"{len(paths)} {label} files, beginning with " + ", ".join(paths[:3]))
    parser.exit(1, "Package contents differ: " + ". ".join(detail) + "\n")
result = {'status':'PASSED','files':len(manifest['files']),'gpu_execution':False}
books = [manifest['book'], manifest['print_pdf'], *manifest.get('compatible_books', [])]
matched = []
for path in args.book:
    digest = checksum(path) if path.is_file() else None
    book = next((entry for entry in books if entry['sha256'] == digest), None)
    if book is None:
        parser.exit(1, f'Book missing or checksum mismatch: {path}\n')
    matched.append(book['name'])
if matched:
    result['matched_books'] = matched
    if len(matched) == 1:
        result['matched_manuscript'] = matched[0]
    if any(name not in (manifest['book']['name'], manifest['print_pdf']['name']) for name in matched):
        result['page_index_for'] = manifest['book']['name']
print(json.dumps(result, indent=2))
