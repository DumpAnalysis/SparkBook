"""Guard the Exercises 29–30 model session and Exercise 251 allocation in WSL2."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from windows_watch import GIB, GuardStop, WindowsWatch, stop_owned, utc

HERE=Path(__file__).resolve().parent


def json_file(path):
    return json.loads(path.read_text())


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def acceptance(events, gib):
    passed=[e for e in events if e.get('event')=='result' and e.get('status')=='PASSED']
    released=any(e.get('event')=='memory' and e.get('stage')=='after_release' for e in events)
    if (len(passed)!=1 or passed[0].get('allocated_written_verified_bytes')!=gib*GIB or
        not released or any(e.get('event')=='release_observation_failed' for e in events) or
        not events or events[-1].get('event')!='process_result' or events[-1].get('exit_code')!=0):
        raise RuntimeError('Byte verification, release or clean process exit is missing')
    return passed[0]


def require_pilot(path,identity):
    receipt=json_file(path/'result.json')
    if (receipt.get('mode')!='pilot' or receipt.get('status')!='PASSED' or
        receipt.get('identity')!=identity or receipt.get('worker_exit')!=0 or receipt.get('host_samples',0)<3):
        raise GuardStop('A passed 1 GiB pilot from this WSL boot and interpreter is required')
    events=[json.loads(s) for s in (path/'stdout.txt').read_text().splitlines()]
    acceptance(events,1)


def stamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def interval(rows, low, high):
    before = [row for at, row in rows if at < low]
    inside = [row for at, row in rows if low <= at <= high]
    after = [row for at, row in rows if at > high]
    return dict(previous=before[-1] if before else None, inside=inside,
                following=after[0] if after else None)


def report(run):
    result = json_file(run/'result.json')
    if result.get('mode') != 'model':
        raise ValueError('A supervised model run is required')
    def optional_json(name, default):
        path = run/name
        return json_file(path) if path.is_file() else default
    requests = optional_json('requests.json', {'requests':[],'placement':None})
    admission = optional_json('admission.json', {})
    def lines(name, encoding='utf-8'):
        path=run/name
        return path.read_text(encoding=encoding).splitlines() if path.is_file() else []
    host = []
    for line in lines('host.jsonl','utf-8-sig'):
        row = json.loads(line)
        host.append((stamp(row['utc']), dict(utc=row['utc'], available_bytes=row['available_bytes'],
                    commit_headroom_bytes=row['commit_limit_bytes']-row['committed_bytes'])))
    guest = []
    since_boot = True
    swap_in = swap_out = None
    for line in lines('vmstat.txt'):
        fields = line.split()
        if 'si' in fields and 'so' in fields:
            swap_in, swap_out = fields.index('si'), fields.index('so')
        elif swap_in is not None and fields and fields[0].isdigit():
            if since_boot:
                since_boot = False
                continue
            at = fields[-2]+'T'+fields[-1]+'+00:00'
            guest.append((stamp(at), dict(utc=at, si=int(fields[swap_in]), so=int(fields[swap_out]))))
    windows=[]
    for row in requests['requests']:
        low,high=stamp(row['started_utc']),stamp(row['ended_utc'])
        windows.append(dict(request=row,host=interval(host,low,high),guest=interval(guest,low,high)))
    pressure=next((w for w in windows if w['request']['label']=='pressure'),None)
    summary=dict(status=result['status'],reason=result.get('reason'),admission=admission,
        failed_checks=[k for k,v in admission.get('checks',{}).items() if not v],
        placement=requests['placement'],completed_requests=len(windows),request_windows=windows,
        pressure_request=pressure['request'] if pressure else None,
        host_window=pressure['host'] if pressure else None,guest_window=pressure['guest'] if pressure else None,
        host_samples=result.get('host_samples',len(host)),guest_file=str(run/'vmstat.txt'),
        host_file=str(run/'host.jsonl'),native_exit=optional_json('native-exit.json',None))
    print(json.dumps(summary,indent=2))
    return {'PASSED':0,'BLOCKED':3,'FAILED':1}[result['status']]


def supervised(args):
    out=args.out.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    mode=args.mode
    is_model=mode=='model'
    result={'mode':mode,'status':'BLOCKED','started_utc':utc(),'worker_exit':None,'host_samples':0}
    worker=vm=None
    try:
        if not Path('/dev/dxg').exists() or 'microsoft' not in Path('/proc/sys/kernel/osrelease').read_text().lower():
            raise GuardStop('WSL2 with a mapped GPU is required for this stage')
        if not shutil.which('vmstat'):raise GuardStop('vmstat is required')
        ident={'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'interpreter':sys.executable,
               'distro':os.environ.get('WSL_DISTRO_NAME')}
        result['identity']=ident
        if mode=='target':require_pilot(args.pilot.expanduser().resolve(),ident)
        floor=(2 if is_model else 4)*GIB
        result['policy']={'windows_floor_bytes':floor,'windows_commit_floor_bytes':floor,'stale_seconds':8,
                          'max_query_seconds':5,'baseline_samples':3,'guest_floor_bytes':2*GIB}
        with WindowsWatch(out,floor) as watch:
            first=watch.baseline()
            argv=[sys.executable,'-B',str(HERE/'model_stage.py'),str(out)] if is_model else [
                sys.executable,'-B',str(HERE/'probe.py'),'--allocate-gib',str(1 if mode=='pilot' else 18),
                '--host-note',json.dumps(first,separators=(',',':'))]
            result['argv']=argv
            result['sources']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ['guarded.py','windows_watch.py','windows_watch.ps1','model_stage.py' if is_model else 'probe.py']}
            with (out/'vmstat.txt').open('xb') as vmout,(out/'vmstat-stderr.txt').open('xb') as vmerr, (out/'stdout.txt').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
                vm=subprocess.Popen(['vmstat','-w','-t','1'],env=dict(os.environ,TZ='UTC',LC_ALL='C'),stdout=vmout,stderr=vmerr,start_new_session=True)
                worker=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
                result['worker_pid']=worker.pid
                deadline=time.monotonic()+(1500 if is_model else 180)
                try:
                    while worker.poll() is None:
                        watch.check()
                        if vm.poll() is not None:raise GuardStop('Guest vmstat stopped')
                        if time.monotonic()>deadline:raise GuardStop('Stage exceeded its time limit')
                        time.sleep(.15)
                    watch.check()
                finally:
                    if worker.poll() is None and not is_model:
                        worker.send_signal(signal.SIGINT)
                        try:worker.wait(timeout=10)
                        except subprocess.TimeoutExpired:pass
                    stop_owned(worker)
                    result['worker_exit']=worker.returncode
                    result['host_samples']=watch.count
                    stop_owned(vm)
                if result['worker_exit']==3:raise GuardStop('Worker prerequisite or admission failed; inspect stderr.txt')
                if result['worker_exit']!=0:raise RuntimeError('Worker failed; inspect its captured output')
            if is_model:
                rows=json_file(out/'requests.json')
                if len(rows['requests'])!=7 or rows['placement']!=[51,65] or json_file(out/'native-exit.json').get('exit') not in (0, -signal.SIGTERM):
                    raise RuntimeError('Expected placement, completions or cleanup missing')
            else:
                events=[json.loads(s) for s in (out/'stdout.txt').read_text().splitlines()]
                result['verification']=acceptance(events,1 if mode=='pilot' else 18)
            result['status']='PASSED'
    except (GuardStop,KeyboardInterrupt) as exc:
        result['status']='BLOCKED';result['reason']=str(exc) or 'Interrupted'
    except Exception as exc:
        result['status']='FAILED';result['reason']=f'{type(exc).__name__}: {exc}'
    finally:
        for process in (worker,vm):
            if process is not None and process.poll() is None:stop_owned(process)
        result['ended_utc']=utc()
        write(out/'result.json',result)
    print(json.dumps(result,indent=2))
    return {'PASSED':0,'BLOCKED':3,'FAILED':1}[result['status']]


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_subparsers(dest='mode',required=True)
    for name in ['model','pilot','target']:
        s=m.add_parser(name);s.add_argument('--out',required=True,type=Path)
        if name=='target':s.add_argument('--pilot',required=True,type=Path)
    s=m.add_parser('report');s.add_argument('--run',required=True,type=Path)
    a=p.parse_args()
    return report(a.run) if a.mode=='report' else supervised(a)


if __name__=='__main__':
    raise SystemExit(main())
