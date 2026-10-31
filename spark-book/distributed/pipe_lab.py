"""Executed process-based teaching control, using pipes rather than PyTorch DDP.

The parent coordinates collectives. This deliberately simple protocol has no
ring, network transport, bucket overlap, fault tolerance, or GPU support.
"""
import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import time
import traceback
import sys
import numpy as np
import torch
from model import make_model, batch, loss, flat, optimizer
from checkpoint_contract import BATCH_RULE, validate_checkpoint


def save(path, data):path.write_text(json.dumps(data,indent=2)+"\n")


def worker(rank, world, conn, out, case, steps, resume):
    sys.stdout=(out/f"stdout-{rank}.txt").open("w",buffering=1)
    sys.stderr=(out/f"stderr-{rank}.txt").open("w",buffering=1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spans=[];sequence=0
    def event(name, **kw):
        with (out/f"rank-{rank}.jsonl").open("a") as f:
            f.write(json.dumps(dict(rank=rank,event=name,monotonic_ns=time.monotonic_ns(),**kw))+"\n")
    @contextmanager
    def span(name, step):
        start=time.monotonic_ns()
        try:yield
        finally:spans.append(dict(rank=rank,name=name,step=step,start_ns=start,end_ns=time.monotonic_ns()))
    def collective(op, values):
        nonlocal sequence
        event("collective_enter",sequence=sequence,operation=op)
        conn.send(dict(rank=rank,sequence=sequence,operation=op,values=values))
        if not conn.poll(30):raise TimeoutError("Worker receive deadline")
        response=conn.recv()
        if "error" in response:raise RuntimeError(response["error"])
        sequence+=1
        event("collective_exit",sequence=sequence-1)
        return response["values"]
    try:
        collective("barrier",None)
        if case in ["crash","mismatch","timeout"]:
            event("fault_boundary",case=case)
            if rank==1 and case=="crash":
                event("injected_exit",code=17)
                raise SystemExit(17)
            if rank==1 and case=="timeout":
                event("injected_delay",seconds=3)
                time.sleep(3)
            collective("broadcast" if rank==1 and case=="mismatch" else "mean",[1.0])
            raise AssertionError("Expected fault was not detected")
        if case=="shard":
            x=np.arange(12,dtype=np.float64).reshape(3,4)/8
            full_w=np.arange(24,dtype=np.float64).reshape(4,6)/16
            w=full_w[:,rank*3:(rank+1)*3].copy()
            parts=collective("gather",(x@w).tolist())
            y=np.concatenate(parts,axis=1)
            dy=np.ones((3,3))
            dx=np.array(collective("sum",(dy@w.T).tolist()))
            dw=x.T@dy
            errors=[float(np.max(np.abs(y-x@full_w))),
                    float(np.max(np.abs(dx-np.ones((3,6))@full_w.T))),
                    float(np.max(np.abs(dw-(x.T@np.ones((3,6)))[:,rank*3:(rank+1)*3])))]
            assert max(errors)<1e-12
            save(out/f"shard-{rank}.json",dict(rank=rank,local_weights=12,full_weights=24,errors=errors))
            return
        model=make_model();opt=optimizer(model);start_step=0
        if resume:
            ck=torch.load(resume,map_location="cpu",weights_only=True)
            validate_checkpoint(ck,world)
            model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["rng"]);start_step=ck["next_step"]
        n=8//world
        event("data_partition",examples=list(range(rank*n,(rank+1)*n)),supervised_tokens=2*n)
        losses=[]
        collective("barrier",None)
        for step in range(start_step,steps):
            if case=="slow" and rank==1 and step==1:
                with span("injected_delay",step):time.sleep(.75)
            ids,labels=batch(step);ids=ids[rank*n:(rank+1)*n];labels=labels[rank*n:(rank+1)*n]
            opt.zero_grad(set_to_none=True)
            with span("forward",step):value=loss(model,ids,labels)
            with span("backward",step):value.backward()
            with span("gradient_average",step):
                grads=torch.cat([p.grad.flatten() for p in model.parameters()])
                averaged=torch.tensor(collective("mean",grads.tolist()),dtype=torch.float64)
                offset=0
                for param in model.parameters():
                    count=param.numel();param.grad.copy_(averaged[offset:offset+count].view_as(param));offset+=count
            with span("optimizer",step):opt.step()
            with span("loss_reduce",step):mean=collective("mean",[float(value.detach())])[0]
            losses.append(mean);event("step_complete",step=step,loss=mean)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),next_step=steps,
            world=world,rng=torch.get_rng_state(),batch_rule=BATCH_RULE),out/f"checkpoint-{rank}.pt")
        save(out/f"model-{rank}.json",dict(parameters=flat(model).tolist(),losses=losses,resumed_at=start_step))
    except BaseException as e:
        event("exception",type=type(e).__name__,message=str(e))
        (out/f"error-{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        save(out/f"spans-{rank}.json",spans);conn.close()


def run(case,out,world=2,steps=4,resume_dir=None):
    out.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context("spawn");parents=[];processes=[]
    for rank in range(world):
        parent,child=ctx.Pipe()
        proc=ctx.Process(target=worker,args=(rank,world,child,out,case,steps,
             resume_dir/f"checkpoint-{rank}.pt" if resume_dir else None))
        proc.start();child.close();parents.append(parent);processes.append(proc)
    pending={};sequence=0;failure=None;active=set(range(world));round_start=time.monotonic()
    deadline=time.monotonic()+75
    try:
        while active:
            now=time.monotonic()
            if now>deadline:raise TimeoutError("Outer coordinator deadline")
            if pending and now-round_start>(1.5 if case=="timeout" and sequence==1 else 25):
                missing=sorted(active-set(pending));failure=dict(kind="timeout",sequence=sequence,missing=missing);break
            ready=wait([parents[r] for r in active if r not in pending],timeout=.05)
            for conn in ready:
                rank=parents.index(conn)
                try:msg=conn.recv()
                except EOFError:
                    active.remove(rank)
                    if case=="crash":failure=dict(kind="worker_exit",rank=rank,sequence=sequence)
                    continue
                if not pending:round_start=time.monotonic()
                pending[rank]=msg
            if failure:break
            if pending and set(pending)==active:
                messages=[pending[r] for r in sorted(active)]
                keys={(m["sequence"],m["operation"]) for m in messages}
                if len(active)!=world or len(keys)!=1 or messages[0]["sequence"]!=sequence:
                    failure=dict(kind="collective_mismatch",sequence=sequence,
                                 observed=[dict(rank=m["rank"],operation=m["operation"],sequence=m["sequence"]) for m in messages]);break
                op=messages[0]["operation"];values=[m["values"] for m in messages]
                if op=="barrier":result=None
                elif op=="gather":result=values
                elif op in ["mean","sum"]:
                    result=np.stack(values).sum(axis=0)
                    if op=="mean":result/=world
                    result=result.tolist()
                else:raise ValueError(op)
                for r in active:parents[r].send(dict(values=result))
                pending={};sequence+=1
        if failure:
            save(out/"failure.json",failure)
            for r in active:
                try:parents[r].send(dict(error=json.dumps(failure)))
                except (BrokenPipeError,OSError):pass
    finally:
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():proc.terminate();proc.join(timeout=2)
            if proc.is_alive():proc.kill();proc.join()
        for c in parents:c.close()
    codes=[p.exitcode for p in processes]
    if case in ["crash","mismatch","timeout"]:
        expected={"crash":"worker_exit","mismatch":"collective_mismatch","timeout":"timeout"}[case]
        ok=bool(failure and failure["kind"]==expected and all(c is not None for c in codes))
        if case=="crash":ok=ok and codes[1]==17
        if case=="timeout":ok=ok and failure["missing"]==[1]
        result=dict(status="EXPECTED_FAILURE" if ok else "FAIL",failure=failure,exitcodes=codes)
        save(out/"result.json",result)
        if not ok:raise AssertionError(result)
        return result
    if failure or any(c!=0 for c in codes):raise AssertionError(dict(failure=failure,exitcodes=codes))
    if case=="shard":
        rows=[json.loads((out/f"shard-{r}.json").read_text()) for r in range(world)]
        result=dict(status="PASS",forward_and_gradient_errors=[r["errors"] for r in rows],local_weights=12,full_weights=24)
    else:
        rows=[json.loads((out/f"model-{r}.json").read_text()) for r in range(world)]
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        reference=make_model();opt=optimizer(reference)
        for step in range(steps):
            opt.zero_grad(set_to_none=True);ids,labels=batch(step);loss(reference,ids,labels).backward();opt.step()
        ref=flat(reference).numpy();actual=np.array(rows[0]["parameters"])
        error=float(np.max(np.abs(actual-ref)))
        replica_error=max(float(np.max(np.abs(np.array(r["parameters"])-actual))) for r in rows)
        result=dict(status="PASS" if error<=1e-9 and replica_error<=1e-9 else "FAIL",
            transport="multiprocessing pipes; parent gradient mean; not PyTorch DDP",
            workers=world,steps=steps,resumed_at=rows[0]["resumed_at"],parameters=len(ref),
            max_reference_error=error,max_replica_error=replica_error,tolerance=1e-9,
            losses=rows[0]["losses"],global_batch=8,masked_tokens=16)
    save(out/"result.json",result)
    if result["status"]!="PASS":raise AssertionError(result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("case",choices=["train","slow","crash","mismatch","timeout","checkpoint","shard"])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=[1,2,4],default=2)
    a=p.parse_args()
    if a.case not in ["train","checkpoint"] and a.workers!=2:raise ValueError("Use two workers for this case")
    if a.case=="checkpoint":
        a.out.mkdir(parents=True,exist_ok=False)
        run("train",a.out/"first",a.workers,steps=2)
        r=run("train",a.out/"resume",a.workers,steps=4,resume_dir=a.out/"first")
        print("checkpoint=PASS next_step=2 model_optimizer_rank_rng")
        print("resume=PASS compared_with_uninterrupted_reference")
    else:r=run(a.case,a.out,a.workers)
    print(f"case={a.case} transport=pipes workers={a.workers}")
    print(f"status={r['status']}")
    if "max_reference_error" in r:
        print(f"parameters={r['parameters']} global_batch=8 masked_tokens=16")
        print(f"max_reference_error={r['max_reference_error']:.3e}")
        print(f"max_replica_error={r['max_replica_error']:.3e}")
    elif "failure" in r:print("diagnosis="+json.dumps(r["failure"],sort_keys=True))
    else:print("local_weights=12 full_weights=24 forward_and_gradients=PASS")


if __name__=="__main__":main()
