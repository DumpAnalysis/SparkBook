"""Export the book's published terminal transcripts as verified UTF-8 files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile

READER = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def paragraph_text(paragraph):
    parts = []
    for element in paragraph.iter():
        if element.tag == W+'t':
            parts.append(element.text or '')
        elif element.tag == W+'tab':
            parts.append('\t')
        elif element.tag in (W+'br', W+'cr'):
            parts.append('\n')
    return ''.join(parts)


def published_captures(book):
    with ZipFile(book) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
    result = {}
    pending = None
    for paragraph in document.find(W+'body'):
        if paragraph.tag != W+'p':
            pending = None
            continue
        text = paragraph_text(paragraph)
        heading = re.match(r'^Recorded run (\d{3}\.\d{2}) — ', text)
        if heading:
            pending = heading[1]
            if pending in result:
                raise ValueError('duplicate recorded run '+pending)
            result[pending] = []
            continue
        if pending is None:
            continue
        style = paragraph.find(W+'pPr/'+W+'pStyle')
        if style is not None and style.get(W+'val') == 'Code':
            result[pending].append(text)
        elif result[pending]:
            pending = None
    return {key:'\n'.join(lines).encode('utf-8') for key,lines in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('book', type=Path)
    parser.add_argument('--out', type=Path, default=READER.parent/'runs/book-transcripts',
                        help='new directory to create in your working lab')
    args = parser.parse_args()
    expected = {row['id']:row['printed_sha256'] for row in
                json.loads((READER/'recorded-runs.json').read_text())['runs']}
    captures = published_captures(args.book)
    if captures.keys() != expected.keys():
        parser.error('the manuscript has different recorded-run IDs')
    for key, data in captures.items():
        if hashlib.sha256(data).hexdigest() != expected[key]:
            parser.error('published text does not match the companion for '+key)
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('choose a new output directory')
    try:
        for key, data in captures.items():
            (args.out/(key+'.txt')).write_bytes(data)
    except BaseException:
        shutil.rmtree(args.out)
        raise
    print(f'Exported {len(captures)} verified transcripts to {args.out.resolve()}')


if __name__ == '__main__':
    main()
