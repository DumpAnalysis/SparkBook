"""CPU preparation workers and one optimizer owner; not distributed training."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from core import model, tokens, write


def make_batch(step):
    ids=list(range(step*8,step*8+8))
    x=[tokens(i) for i in ids]
    y=[[((i*7+j*3)%31-15)/100 for j in range(64)] for i in ids]
    return x,y


def prepare(conn):
    conn.send({'pid':os.getpid(),'role':'CPU preparation'})
    while True:
        step=conn.recv()
        if step is None:break
        conn.send({'step':step,'data':make_batch(step),'pid':os.getpid()})
    conn.close()


def execute(device,out):
    import torch
    torch.set_num_threads(1)
    ctx=mp.get_context('spawn');procs=[];pipes=[]
    try:
        for _ in range(2):
            a,b=ctx.Pipe();p=ctx.Process(target=prepare,args=(b,))
            p.start();b.close();procs.append(p);pipes.append(a)
        inventory=[]
        for a in pipes:
            if not a.poll(30):raise TimeoutError('preparation startup')
            inventory.append(a.recv())
        m=model(device).train();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        losses=[];events=[]
        # Two bounded outstanding batches. Main process preserves step order.
        for step in range(2):pipes[step].send(step)
        for step in range(4):
            a=pipes[step%2]
            if not a.poll(30):raise TimeoutError('preparation batch')
            item=a.recv()
            if item['step']!=step:raise AssertionError('data order')
            if step+2<4:a.send(step+2)
            x,y=item['data']
            start=time.perf_counter_ns()
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(m(torch.tensor(x,device=device)),
                                             torch.tensor(y,device=device))
            loss.backward();opt.step()
            value=float(loss.detach().cpu());losses.append(value)
            events.append(dict(step=step,preparer=item['pid'],trainer=os.getpid(),
                               start_ns=start,end_ns=time.perf_counter_ns()))
        actual={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        counters=sorted(set(int(s['step'].item()) for s in opt.state.values()))
        torch.save({'model':actual,'optimizer':opt.state_dict(),'next_step':4},
                   out/'training-checkpoint.pt')
        del opt,m
        ref=model('cpu').train();ro=torch.optim.AdamW(ref.parameters(),lr=.001)
        reference_losses=[]
        for step in range(4):
            x,y=make_batch(step);ro.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(ref(torch.tensor(x)),torch.tensor(y))
            loss.backward();ro.step();reference_losses.append(float(loss.detach()))
        error=max(float((actual[k]-v).abs().max()) for k,v in ref.state_dict().items())
        loss_error=max(abs(a-b) for a,b in zip(losses,reference_losses))
        if error>1e-5 or loss_error>1e-6 or counters!=[4]:
            raise AssertionError('training reference disagreement')
        for a in pipes:a.send(None)
        for p in procs:
            p.join(10)
            if p.exitcode!=0:raise RuntimeError('preparation process failed')
        result=dict(status='PASS',device=device,preparers=inventory,trainer=os.getpid(),
          losses=losses,reference_losses=reference_losses,max_abs_error=error,
          loss_max_abs_error=loss_error,optimizer_steps=counters,events=events,
          next_step=4,training_model_owners=1)
        write(out/'result.json',result)
        print(f'PASS device={device} CPU_preparers=2 training_owners=1 updates=4')
        print('losses='+','.join(f'{v:.8f}' for v in losses))
        print(f'reference_max_abs_error={error:.3e} optimizer_steps={counters}')
        print('checkpoint=training-checkpoint.pt next_step=4')
    finally:
        for p in procs:
            if p.is_alive():p.terminate()
        for p in procs:
            p.join(2)
            if p.is_alive():p.kill();p.join(2)
        for a in pipes:a.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    import torch
    if a.device=='cuda' and not torch.cuda.is_available():
        write(out/'result.json',dict(status='UNAVAILABLE',reason='CUDA unavailable'))
        print('UNAVAILABLE CUDA unavailable; no GPU run executed');return 2
    execute(a.device,out);return 0


if __name__=='__main__':sys.exit(main())
