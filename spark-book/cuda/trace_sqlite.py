"""Inspect only a named Nsight Systems SQLite export, read-only."""
import argparse,pathlib,sqlite3
TABLES=['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY',
        'CUPTI_ACTIVITY_KIND_RUNTIME','StringIds']

def summarize(path):
    p=pathlib.Path(path).resolve()
    with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as con:
        present={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        available=[x for x in TABLES if x in present]
        if not {'CUPTI_ACTIVITY_KIND_KERNEL','StringIds'}<=present:
            raise ValueError('Expected kernel/StringIds tables absent; inspect export schema')
        columns={r[1] for r in con.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        string_columns={r[1] for r in con.execute('PRAGMA table_info(StringIds)')}
        if not {'start','end','shortName'}<=columns or not {'id','value'}<=string_columns:
            raise ValueError('Export schema differs; inspect before adapting query')
        if con.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end<start').fetchone()[0]:
            raise ValueError('Negative event duration')
        unmatched=con.execute('''SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds s ON k.shortName=s.id WHERE s.id IS NULL''').fetchone()[0]
        if unmatched:raise ValueError('Kernel name IDs missing from StringIds')
        rows=con.execute('''SELECT s.value,count(*),sum(k.end-k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
            GROUP BY s.value ORDER BY s.value''').fetchall()
        return available,rows

def fixture(path):
    p=pathlib.Path(path)
    if p.exists():raise ValueError('Use a new fixture path')
    p.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(p) as con:
        con.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER);
          INSERT INTO StringIds VALUES(1,'transform');
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,300,1),(400,700,1);''')

def main():
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--fixture',action='store_true')
    args=p.parse_args()
    if args.fixture:fixture(args.path)
    tables,rows=summarize(args.path)
    print('evidence='+('CPU-created schema fixture; no GPU trace' if args.fixture else 'supplied SQLite export'))
    print('tables='+','.join(tables))
    for name,count,ns in rows:print(f'kernel={name} instances={count} summed_duration_ns={ns}')
    print('Summed kernel duration is not application wall time.')
if __name__=='__main__':main()
