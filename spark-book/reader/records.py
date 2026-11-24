"""Find printed run pages, published transcripts and local exercise files."""
import argparse
import json
from pathlib import Path

READER = Path(__file__).resolve().parent
ROOT = READER.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('exercise', type=int)
    p.add_argument('--files', action='store_true', help='show paths within this companion')
    p.add_argument('--transcripts', type=Path, default=ROOT/'runs/book-transcripts',
                   help='directory created by reader/import_transcripts.py')
    args = p.parse_args()
    guide = ROOT/f'reader/exercises/{args.exercise:03}.txt'
    if not 1 <= args.exercise <= 251 or not guide.is_file():
        p.error('exercise must be in 1..251')
    entries = json.loads((ROOT/'EXERCISE_EVIDENCE_MAP.json').read_text())['exercises']
    entry = next(x for x in entries if x['number'] == args.exercise)
    runs = json.loads((READER/'recorded-runs.json').read_text())['runs']
    print(f"Exercise {args.exercise}  {entry['title']} (book p. {entry['book_page']})")
    print(entry['evidence_note'])
    selected = [row for row in runs if row['exercise'] == args.exercise]
    if entry.get('record_locator'):
        print(entry['record_locator'])
    shared_ids = set(entry.get('shared_record_ids', []))
    selected.extend(row for row in runs if row['id'] in shared_ids)
    for number in entry.get('shared_capture_exercises', []):
        print(f'Shared captures: python3 reader/records.py {number}')
    for row in selected:
        print(f"\nRecorded run {row['id']} — {row['title']} (book p. {row['book_page']})")
        print(row['label'])
        print(row['scope'])
        print('Read first: '+row['read_first'])
        if args.files:
            transcript = args.transcripts/(row['id']+'.txt')
            if transcript.is_file():
                print('Published transcript: '+str(transcript.resolve()))
    if args.files:
        print('\nLocal guide: '+str(guide))
        for path in entry.get('companion_files', []):
            print('Exercise file: '+str(ROOT/path))
        if selected and not args.transcripts.is_dir():
            print('For verified plain text, run python3 reader/import_transcripts.py in your working lab.')
    print('This lookup does not execute the experiment.')


if __name__ == '__main__':
    main()
