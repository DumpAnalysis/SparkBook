"""Display exercise records and optionally locate their unchanged evidence files."""
import argparse,json,os
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('exercise',type=int)
    p.add_argument('--files',action='store_true',help='locate files under SPARK_BOOK_EVIDENCE')
    a=p.parse_args()
    guides=json.loads((R/'exercise-guide.json').read_text())['exercises']
    matches=[r for r in guides if r['number']==a.exercise]
    if not matches:p.error('exercise must be in 1..251')
    g=matches[0]
    runs=[r for r in json.loads((R/'recorded-runs.json').read_text())['runs'] if r['exercise']==a.exercise]
    root=None
    if a.files:
        env=os.environ.get('SPARK_BOOK_EVIDENCE')
        if not env:p.error('set SPARK_BOOK_EVIDENCE to the restored evidence directory; see reader/EVIDENCE.txt')
        root=Path(env).expanduser().resolve()
        if not (root/'exercise-aliases.json').is_file():p.error('restore the matching Version 48 evidence pack first')
    print(f"Exercise {a.exercise}: {g['title']}\n\n{g['evidence_note']}\n")
    if not runs:
        print('No separate printed capture is indexed here. Read the exercise and supporting records.')
        index=json.loads((R.parent/'EXERCISE_EVIDENCE_MAP.json').read_text())['exercises']
        entry=next(row for row in index if row['number']==a.exercise)
        for n in entry.get('shared_capture_exercises',[]):print(f'Shared capture: python reader/records.py {n}')
        print()
    for r in runs:
        print(f"Recorded run {r['id']} — {r['title']}\n{r['label']}\n{r['scope']}\nRead first: {r['read_first']}")
        if root:
            for name,row in r['files'].items():
                f=root/row['path']
                if not f.is_file():p.error('missing record file: '+str(f))
                print(f'  {name}: {f}')
        print()
    if root:
        exroot=root/'exercises'/f'{a.exercise:03d}'
        for name in ['assessments','resources','source-index.json']:
            f=exroot/name
            if f.exists():print(f'{name}: {f}')
    else:print('Use --files after restoring the evidence pack. See reader/EVIDENCE.txt.')
    print('These references display existing observations; no experiment was executed.')
if __name__=='__main__':main()
