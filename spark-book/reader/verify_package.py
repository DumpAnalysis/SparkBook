"""Check the pristine learner package and optionally its matching book."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--book', type=Path, help='verify the original, unedited DOCX file')
args = parser.parse_args()
name = 'VERSION56_MANIFEST.json'
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
    parser.exit(1, 'Unexpected or missing files. Check a pristine extraction.\n')
result = {'status':'PASSED','files':len(manifest['files']),'gpu_execution':False}
if args.book:
    books = [manifest['book'], *manifest.get('compatible_books', [])]
    digest = checksum(args.book) if args.book.is_file() else None
    match = next((book for book in books if book['sha256'] == digest), None)
    if match is None:
        parser.exit(1, 'Manuscript missing or checksum mismatch. Expected an unedited matching book.\n')
    result['matched_manuscript'] = match['name']
    if match != manifest['book']:
        result['page_index_for'] = manifest['book']['name']
print(json.dumps(result, indent=2))
