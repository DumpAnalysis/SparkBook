"""Inspect a named Nsight Systems SQLite export in read-only mode."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

KERNEL = 'CUPTI_ACTIVITY_KIND_KERNEL'
COPY = 'CUPTI_ACTIVITY_KIND_MEMCPY'
TABLES = [KERNEL, COPY, 'CUPTI_ACTIVITY_KIND_RUNTIME', 'StringIds']
VALID = 'start > 0 AND end > start'


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def summarize(path):
    with closing(connect(path)) as con:
        present = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {KERNEL, 'StringIds'} <= present:
            raise ValueError('Expected kernel/StringIds tables absent; device duration unavailable')
        columns = {r[1] for r in con.execute(f'PRAGMA table_info({KERNEL})')}
        names = {r[1] for r in con.execute('PRAGMA table_info(StringIds)')}
        if not {'start', 'end', 'shortName'} <= columns or not {'id', 'value'} <= names:
            raise ValueError('Export schema differs; inspect before adapting query')
        if con.execute(f'SELECT 1 FROM {KERNEL} WHERE end < start LIMIT 1').fetchone():
            raise ValueError('Negative event duration')
        if con.execute(f'''SELECT 1 FROM {KERNEL} k LEFT JOIN StringIds s ON k.shortName=s.id
                           WHERE s.id IS NULL LIMIT 1''').fetchone():
            raise ValueError('Kernel name IDs missing from StringIds')
        rows = con.execute(f'''SELECT s.value, count(*), sum(end-start) FROM {KERNEL} k
                               JOIN StringIds s ON k.shortName=s.id WHERE {VALID}
                               GROUP BY s.value ORDER BY s.value''').fetchall()
        if not rows:
            raise ValueError('No positive kernel intervals; device duration unavailable')
        return [name for name in TABLES if name in present], rows


def stream_check(con, match, pair, tables):
    # Look up the full demangled name, since a short name can be simply "Kernel2".
    keys = {r[1] for r in con.execute(f'PRAGMA table_info({KERNEL})')}
    if not {'streamId', 'demangledName', 'deviceId'} <= keys:
        raise ValueError('Stream or demangled-name fields unavailable')
    selected = con.execute(f'''SELECT start,end,streamId,deviceId FROM {KERNEL} k
        JOIN StringIds s ON s.id=k.demangledName WHERE {VALID}
        AND instr(lower(s.value),lower(?))>0 ORDER BY start''', (match,)).fetchall()
    if not selected:
        raise ValueError(f'No timed kernel matches {match!r}')
    print('selected_kernel_streams=' + str(dict(Counter(x[2] for x in selected))))
    if pair == 'copies':
        if COPY not in tables:
            raise ValueError('Copy activity table absent; overlap unavailable')
        others = con.execute(f'SELECT start,end,streamId,deviceId FROM {COPY} WHERE {VALID} ORDER BY start').fetchall()
        if not others:
            raise ValueError('No positive copy intervals; overlap unavailable')
        print(f'timed_copies={len(others)}')
    else:
        others = selected
    pairs = 0
    for i, (start, end, stream, device) in enumerate(selected):
        for j, (left, right, other_stream, other_device) in enumerate(others):
            if left >= end:
                break
            if (pair == 'copies' or j > i) and other_device == device and other_stream != stream and right > start:
                pairs += 1
    print(f'cross_stream_overlap_pairs={pairs}')


def request_check(con, path):
    receipt = json.loads(path.read_text())
    epoch, = con.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()
    rows = con.execute(f'''SELECT p.name,count(*) FROM {KERNEL} k JOIN PROCESSES p ON p.globalPid=k.globalPid
        WHERE p.pid=? AND start>=? AND end<=? AND {VALID} GROUP BY p.name''',
        (receipt['pid'], receipt['start_utc_ns'] - epoch, receipt['end_utc_ns'] - epoch)).fetchall()
    expected = receipt['command'][0]
    count = sum(n for name, n in rows if name in (expected, Path(expected).name))
    print(f"owner_pid={receipt['pid']} request_window_kernels={count}")
    if not count:
        raise ValueError('No matching owner kernel entirely inside the request window')


def ranges_check(con, prefix):
    rows = con.execute('''SELECT coalesce(n.text,s.value),start,end FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id=n.textId WHERE end>start ORDER BY start''')
    found = []
    for name, start, end in rows:
        if name and name.startswith(prefix):
            count, = con.execute(f'SELECT count(*) FROM {KERNEL} WHERE {VALID} AND start>=? AND end<=?',
                                 (start, end)).fetchone()
            print(f'range={name} contained_kernels={count}')
            found.append(count)
    if not found:
        raise ValueError('No matching NVTX ranges')


def launch_check(con, prefix):
    # Nsight encodes the thread ID in the low 24 bits of globalTid.
    rows = con.execute("""SELECT coalesce(n.text, label.value), rt.correlationId,
           kernel.value, k.start, k.end, n.start, n.end
        FROM NVTX_EVENTS n LEFT JOIN StringIds label ON label.id=n.textId
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME rt ON rt.globalTid=n.globalTid
            AND rt.start>=n.start AND rt.end<=n.end
        JOIN StringIds api ON api.id=rt.nameId
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=rt.correlationId
            AND k.globalPid=(rt.globalTid & ~16777215) AND k.start>0 AND k.end>k.start
        JOIN StringIds kernel ON kernel.id=k.demangledName
        WHERE substr(coalesce(n.text,label.value),1,length(?))=? AND api.value LIKE '%LaunchKernel%'
        ORDER BY n.start,rt.start,k.start""", (prefix,prefix)).fetchall()
    for name,corr,kernel,start,end,left,right in rows:
        print(f'range={name} launch_id={corr} kernel={kernel} wholly_contained={left<=start and end<=right}')
    print(f'correlated_launches={len(rows)}')
    if not rows:raise ValueError('No device kernels correlated with a launch in the selected ranges')


def fixture(path):
    p = Path(path)
    if p.exists():
        raise ValueError('Use a new fixture path')
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        con.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER);
          INSERT INTO StringIds VALUES(1,'transform');
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,300,1),(400,700,1);''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--match', help='substring in the full kernel name, for a cross-stream overlap check')
    p.add_argument('--overlap', choices=('kernels', 'copies'), default='kernels')
    p.add_argument('--request', type=Path, help='request.json from trace_server.py')
    p.add_argument('--ranges', help='list NVTX ranges starting with this prefix and their contained device events')
    p.add_argument('--launches',help='correlate in-range CUDA launches with kernels, including kernels outside the host range')
    args = p.parse_args()
    if args.fixture:
        fixture(args.path)
    try:
        tables, rows = summarize(args.path)
        print('evidence=' + ('CPU-created schema fixture; no GPU trace' if args.fixture else 'supplied SQLite export'))
        print('tables=' + ','.join(tables))
        for name, count, ns in rows:
            print(f'kernel={name} instances={count} summed_duration_ns={ns}')
        print('Summed kernel duration is not application wall time.')
        with closing(connect(args.path)) as con:
            if args.match is not None:
                stream_check(con, args.match, args.overlap, tables)
            if args.request:
                request_check(con, args.request)
            if args.launches:
                launch_check(con, args.launches)
            if args.ranges:
                ranges_check(con, args.ranges)
    except (ValueError, sqlite3.Error) as exc:
        p.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
