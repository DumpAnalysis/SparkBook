"""Bounded local process service; one outstanding request per CPU client.

All IPC payloads are Python values. No CUDA tensors cross process boundaries.
One owner per shared service, or one independent owner per client in replicas
mode. All CUDA owners target cuda:0; there are no distributed/NCCL ranks.
"""
import argparse
from collections import deque
import contextlib
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from core import model, tokens, oracle, memory, write


def client(conn, rank, workers, requests, out):
    events = []
    try:
        conn.send({'kind': 'ready', 'pid': os.getpid()})
        assert conn.recv() == 'go'
        for rid in range(rank, requests, workers):
            t0 = time.perf_counter_ns()
            conn.send({'kind': 'request', 'id': rid, 'client': rank,
                       'tokens': tokens(rid), 'sent_ns': t0})
            response = conn.recv()
            t1 = time.perf_counter_ns()
            if response.get('kind') != 'result' or response['id'] != rid:
                raise RuntimeError('wrong response identity')
            events.append(dict(id=rid, client=rank, sent_ns=t0,
                               received_ns=t1, value=response['value']))
        conn.send({'kind': 'done'})
        write(Path(out) / f'client-{rank}.json', events)
    except BaseException:
        (Path(out) / f'client-{rank}.stderr').write_text(traceback.format_exc())
        raise
    finally:
        conn.close()


def owner(conn, index, device, out, profile):
    import torch
    path = Path(out)
    try:
        m = model(device)
        with torch.inference_mode():
            m(torch.tensor([tokens(0)], device=device)).cpu()
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        ctx = (torch.profiler.profile(activities=activities,
                                     record_shapes=True, profile_memory=True)
               if profile else contextlib.nullcontext())
        with ctx as prof:
            conn.send(dict(kind='ready', pid=os.getpid(), device=device,
                           model_bytes=sum(p.numel()*p.element_size()
                                           for p in m.parameters()),
                           memory=memory(device)))
            while True:
                msg = conn.recv()
                if msg['kind'] == 'stop':
                    break
                t0 = time.perf_counter_ns()
                with torch.inference_mode(), torch.profiler.record_function('shared_batch'):
                    x = torch.tensor([r['tokens'] for r in msg['requests']],
                                     dtype=torch.long, device=device)
                    values = m(x).cpu().tolist()
                t1 = time.perf_counter_ns()
                conn.send(dict(kind='batch', owner=index,
                               ids=[r['id'] for r in msg['requests']],
                               values=values, start_ns=t0, end_ns=t1,
                               memory=memory(device)))
        if profile:
            prof.export_chrome_trace(str(path / f'owner-{index}-profiler.json'))
        write(path / f'owner-{index}.json',
              dict(pid=os.getpid(), device=device, memory=memory(device)))
    except BaseException:
        (path / f'owner-{index}.stderr').write_text(traceback.format_exc())
        raise
    finally:
        conn.close()


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    write(out / 'config.json', vars(args) | dict(python=sys.version,
          torch=torch.__version__, cuda_runtime=torch.version.cuda,
          start_method='spawn', outstanding_per_client=1))
    if args.device == 'cuda' and not torch.cuda.is_available():
        write(out / 'result.json', {'status':'UNAVAILABLE', 'reason':'CUDA unavailable'})
        print('UNAVAILABLE CUDA unavailable; no GPU run executed')
        return 2
    ctx = mp.get_context('spawn')
    procs, cp, op = [], [], []
    count = 1 if args.mode == 'shared' else args.workers
    started = time.perf_counter()
    deadline = started + 90
    try:
        for i in range(count):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=owner, args=(child,i,args.device,str(out),args.profile))
            p.start(); child.close(); procs.append(p); op.append(parent)
        for i in range(args.workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=client,args=(child,i,args.workers,args.requests,str(out)))
            p.start(); child.close(); procs.append(p); cp.append(parent)
        ready = {}; allpipes = op + cp
        while len(ready) < len(allpipes):
            if time.perf_counter() > deadline: raise TimeoutError('startup deadline')
            for c in wait([c for c in allpipes if c not in ready], .05):
                msg = c.recv()
                if msg['kind'] != 'ready': raise RuntimeError('startup protocol')
                ready[c] = msg
        owners = [ready[c] for c in op]
        begin = time.perf_counter_ns()
        for c in cp: c.send('go')
        queues = [deque() for _ in cp]
        busy = [None] * count
        live = set(range(args.workers))
        batches, requests = [], []
        cursor, max_pending = 0, 0
        while live or any(busy) or any(queues):
            if time.perf_counter() > deadline: raise TimeoutError('run deadline')
            watched = [cp[i] for i in live] + [op[i] for i,b in enumerate(busy) if b]
            for c in wait(watched, .001) if watched else []:
                msg = c.recv()
                if c in cp:
                    i = cp.index(c)
                    if msg['kind'] == 'done': live.remove(i)
                    elif msg['kind'] == 'request':
                        if queues[i] or any(b and any(r['client']==i for r in b) for b in busy):
                            raise RuntimeError('one outstanding request contract')
                        msg['accepted_ns'] = time.perf_counter_ns()
                        queues[i].append(msg)
                    else: raise RuntimeError('client protocol')
                else:
                    oi = op.index(c); batch = busy[oi]
                    if msg['ids'] != [r['id'] for r in batch]: raise RuntimeError('batch identity')
                    batches.append(msg)
                    for r,value in zip(batch,msg['values']):
                        r = dict(r); r.update(owner=oi, start_ns=msg['start_ns'], end_ns=msg['end_ns'])
                        requests.append(r)
                        cp[r['client']].send(dict(kind='result',id=r['id'],value=value))
                    busy[oi] = None
            max_pending = max(max_pending,sum(map(len,queues)))
            for oi in range(count):
                if busy[oi]: continue
                eligible = list(range(args.workers)) if count == 1 else [oi]
                total = sum(len(queues[i]) for i in eligible)
                if not total: continue
                oldest = min(queues[i][0]['accepted_ns'] for i in eligible if queues[i])
                age = (time.perf_counter_ns()-oldest)/1e6
                target = min(args.batch,len(eligible))
                if total < target and age < args.window_ms: continue
                batch = []
                # Round-robin across clients; a batch gets at most one per client.
                for offset in range(args.workers):
                    i = (cursor+offset) % args.workers
                    if i in eligible and queues[i] and len(batch) < args.batch:
                        batch.append(queues[i].popleft())
                cursor = (batch[-1]['client']+1) % args.workers
                op[oi].send(dict(kind='batch',requests=batch)); busy[oi]=batch
        end = time.perf_counter_ns()
        for c in op: c.send({'kind':'stop'})
        for p in procs:
            p.join(max(0,deadline-time.perf_counter()))
            if p.exitcode != 0: raise RuntimeError(f'child {p.pid} exit={p.exitcode}')
        events = []
        for i in range(args.workers):
            events += __import__('json').loads((out/f'client-{i}.json').read_text())
        events.sort(key=lambda r:r['id'])
        if [r['id'] for r in events] != list(range(args.requests)):
            raise AssertionError('missing or duplicate request')
        actual=np.asarray([r['value'] for r in events])
        expected=oracle([tokens(i) for i in range(args.requests)])
        error=float(np.max(np.abs(actual-expected)))
        if error > 1e-6: raise AssertionError(f'oracle error {error}')
        lat=[(r['received_ns']-r['sent_ns'])/1e6 for r in events]
        queue=[(r['start_ns']-r['accepted_ns'])/1e6 for r in requests]
        result=dict(status='PASS',device=args.device,workers=args.workers,
          owners=count,requests=len(events),batches=len(batches),
          max_pending=max_pending,model_bytes=sum(r['model_bytes'] for r in owners),
          max_abs_error=error,latency_p50_ms=float(np.percentile(lat,50)),
          latency_p95_ms=float(np.percentile(lat,95)),
          queue_p95_ms=float(np.percentile(queue,95)),
          elapsed_ms=(end-begin)/1e6,requests_per_second=len(events)/((end-begin)/1e9),
          owners_inventory=owners,child_exitcodes=[p.exitcode for p in procs])
        write(out/'batches.json',batches);write(out/'requests.json',requests)
        write(out/'result.json',result)
        print(f'PASS {args.mode} device={args.device} clients={args.workers} owners={count}')
        print(f"requests={len(events)} batches={len(batches)} model_bytes={result['model_bytes']}")
        print(f'max_pending={max_pending} max_abs_error={error:.3e}')
        print(f"latency_ms p50={result['latency_p50_ms']:.3f} p95={result['latency_p95_ms']:.3f}")
        print(f"queue_p95_ms={result['queue_p95_ms']:.3f} requests_per_second={result['requests_per_second']:.2f}")
        print('CUDA allocator metrics=' + ('recorded per owner' if args.device=='cuda' else 'unavailable on CPU'))
        return 0
    except BaseException:
        (out/'driver.stderr').write_text(traceback.format_exc())
        write(out/'result.json',dict(status='FAILED',error=traceback.format_exc()))
        raise
    finally:
        # Bound cleanup to children of this invocation. Never touch another job.
        for p in procs:
            if p.is_alive(): p.terminate()
        for p in procs:
            p.join(2)
            if p.is_alive(): p.kill();p.join(2)
        for c in cp+op:c.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['shared','replicas'],default='shared')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--workers',type=int,choices=[1,2,4],default=4)
    p.add_argument('--requests',type=int,default=16)
    p.add_argument('--batch',type=int,choices=[1,2,4],default=4)
    p.add_argument('--window-ms',type=float,default=5)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if not args.workers <= args.requests <= 1024: p.error('requests must be workers..1024')
    if not 0 <= args.window_ms <= 1000: p.error('window-ms must be 0..1000')
    return run(args)


if __name__=='__main__':sys.exit(main())
