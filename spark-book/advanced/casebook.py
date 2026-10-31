"""Small deterministic diagnostic fixtures, not captured CUDA incidents."""
import argparse
import json


def cases():
    # Each fault has a negative control and an explicit independent oracle.
    x = list(range(15))
    expected = [x[r * 5 + c] for c in range(5) for r in range(3)]
    wrong = [x[r * 3 + c] for c in range(5) for r in range(3)]
    yield ('stride', wrong != expected, expected == [0,5,10,1,6,11,2,7,12,3,8,13,4,9,14],
           'row stride must be 5; square-only tests concealed the fault')
    prior, wanted = [2, 2, 2], [4, 6, 8]
    yield ('stale', prior != wanted, [2 * v for v in [2,3,4]] == wanted,
           'output from the preceding request survived; poison then verify every element')
    def schedule(order):
        value=0; output=None
        for action in order:
            if action=='produce':value=1
            else:output=2*value+1
        return output
    yield ('dependency', schedule(['consume','produce']) != 3,
           schedule(['produce','consume']) == 3,
           'consumer needs an event recorded after this producer submission')
    live, reserved = [4,4,4], [8,12,12]
    yield ('cache', reserved[-1] > reserved[0], len(set(live)) == 1,
           'reserved growth alone does not prove live tensor retention or fragmentation')
    retained = [[0] * 8 for _ in range(3)]
    held = sum(len(a) for a in retained)
    retained.clear()
    yield ('retention', held == 24, len(retained) == 0,
           'a Python owner kept outputs alive; detach alone would still retain their storage')
    events=[('launch','badwrite'),('fault','badwrite'),('report','synchronize')]
    first=next(name for kind,name in events if kind=='fault')
    reported=next(name for kind,name in events if kind=='report')
    yield ('delayed', first != reported, first == 'badwrite',
           'the reporting API is not necessarily the first failing kernel')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    rows = []
    if not a.json:
        print('evidence=CPU teaching fixtures; no CUDA acquisition')
    for name, fault, repair, conclusion in cases():
        assert fault and repair
        rows.append(dict(case=name, fault_exposed=fault,
                         repair_or_interpretation_checked=repair,
                         conclusion=conclusion))
        if not a.json:
            print(f'{name}: fault_exposed=true check=PASS')
            print('  ' + conclusion)
    if a.json:
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
