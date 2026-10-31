"""CPU-only service control plane faults. No active CUDA context is killed."""
import argparse
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import time
from core import oracle,tokens,write


class Pending:
    def __init__(self,capacity):self.capacity=capacity;self.q=deque()
    def submit(self,request):
        if any(r['id']==request['id'] for r in self.q):return 'DUPLICATE'
        if len(self.q)>=self.capacity:return 'BUSY'
        self.q.append(request);return 'ACCEPTED'
    def cancel(self,rid):
        for r in list(self.q):
            if r['id']==rid:self.q.remove(r);return 'CANCELLED'
        return 'NOT_PENDING'
    def take(self,now):
        r=self.q.popleft()
        return r,('EXPIRED' if now>=r['deadline'] else 'DISPATCH')


def owner(conn,crash):
    conn.send('ready')
    request=conn.recv()
    if crash:os._exit(17)  # Before any tensor/GPU computation or reply.
    conn.send(dict(id=request['id'],value=oracle([tokens(request['id'])])[0].tolist()))
    conn.close()


def attempt(crash):
    ctx=mp.get_context('spawn');a,b=ctx.Pipe()
    p=ctx.Process(target=owner,args=(b,crash));p.start();b.close()
    result=None
    try:
        if not a.poll(10) or a.recv()!='ready':raise TimeoutError('owner startup')
        a.send({'id':7})
        if not a.poll(5):raise TimeoutError('response deadline')
        try:result=a.recv()
        except EOFError:result={'status':'OWNER_LOST','outcome':'UNKNOWN'}
        p.join(5)
        if p.is_alive():raise TimeoutError('owner shutdown')
        return dict(response=result,exitcode=p.exitcode)
    finally:
        if p.is_alive():p.terminate();p.join(2)
        if p.is_alive():p.kill();p.join(2)
        a.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    # Explicit logical times make deadline handling deterministic.
    q=Pending(2);events=[]
    for rid in [1,2,3]:events.append([rid,q.submit(dict(id=rid,deadline=10))])
    events.append([1,q.cancel(1)])
    events.append([3,q.submit(dict(id=3,deadline=20))])
    r,status=q.take(now=11);events.append([r['id'],status])
    r,status=q.take(now=11);events.append([r['id'],status])
    events.append([3,q.cancel(3)])
    failed=attempt(True);recovered=attempt(False)
    expected=[[1,'ACCEPTED'],[2,'ACCEPTED'],[3,'BUSY'],[1,'CANCELLED'],
              [3,'ACCEPTED'],[2,'EXPIRED'],[3,'DISPATCH'],[3,'NOT_PENDING']]
    assert events==expected
    assert failed['exitcode']==17 and failed['response']['status']=='OWNER_LOST'
    assert recovered['exitcode']==0 and recovered['response']['id']==7
    write(out/'result.json',dict(status='PASS',queue_events=events,
          failed=failed,recovered=recovered,scope='CPU control plane'))
    print('PASS CPU control plane capacity=2')
    print('admission=ACCEPTED,ACCEPTED,BUSY')
    print('pending_cancel=CANCELLED expired=EXPIRED dispatched_cancel=NOT_PENDING')
    print('owner_exit=17 request_outcome=UNKNOWN')
    print('fresh_owner_exit=0 read_only_request=7 completed')


if __name__=='__main__':main()
