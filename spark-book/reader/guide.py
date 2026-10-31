"""Display the learner guide without executing its commands."""
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('exercise', type=int, nargs='?')
parser.add_argument('--list', action='store_true')
args = parser.parse_args()
if args.list:
    for file in sorted((root / 'exercises').glob('*.txt')):
        _, number, title = file.read_text().splitlines()[0].split(maxsplit=2)
        print(f'{int(number):03d}  {title}')
elif args.exercise is None or not 1 <= args.exercise <= 251:
    parser.error('supply an exercise number in 1..251 or --list')
else:
    print((root / 'exercises' / f'{args.exercise:03}.txt').read_text(), end='')
