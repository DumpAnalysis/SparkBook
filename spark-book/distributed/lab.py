"""Bounded, local-only distributed laboratory. All child output is retained."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(path, value): path.write_text(json.dumps(value, indent=2)+"\n")


def capture(cmd, out, timeout=90, debug_level="INFO"):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo",
               TORCH_DISTRIBUTED_DEBUG=debug_level, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    start = time.monotonic()
    with (out/"stdout.txt").open("w") as stdout, (out/"stderr.txt").open("w") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        expired = False
        try: rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            rc = proc.returncode
        finally:
            # Clean any surviving descendants of this owned launcher only.
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    record = dict(argv=cmd, command=shlex.join(cmd), returncode=rc,
                  timed_out=expired, seconds=time.monotonic()-start,
                  controlled_environment={k:env[k] for k in
                  ["OMP_NUM_THREADS","MKL_NUM_THREADS","GLOO_SOCKET_IFNAME",
                   "TORCH_DISTRIBUTED_DEBUG","CUBLAS_WORKSPACE_CONFIG"]})
    save(out/"launch.json", record)
    return record


def launch(out, mode, workers=2, args=(), debug_level="INFO"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
    # Static rendezvous also keeps the worker store on explicit loopback.
    # Port selection has a bind-to-launch race; failures remain in raw logs.
    cmd = [sys.executable,"-m","torch.distributed.run","--master-addr=127.0.0.1",
           f"--master-port={port}","--rdzv-conf=timeout=45","--nnodes=1",f"--nproc-per-node={workers}",
           "--max-restarts=0",str(ROOT/"worker.py"),mode,"--out",str(out),*args]
    return capture(cmd, out, debug_level=debug_level)


def probe(out):
    import torch
    import torch.distributed as dist
    data = dict(python=platform.python_version(), machine=platform.machine(),
        kernel=platform.release(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, distributed=dist.is_available(),
        gloo=dist.is_gloo_available(), nccl=dist.is_nccl_available(),
        cuda_devices=torch.cuda.device_count(), devices=[])
    for i in range(data["cuda_devices"]):
        p=torch.cuda.get_device_properties(i)
        data["devices"].append(dict(index=i,name=p.name,total_memory_bytes=p.total_memory,
            capability=[p.major,p.minor], uuid=str(getattr(p,"uuid","unavailable"))))
    save(out/"probe.json",data)
    for k in ["python","machine","torch","torch_cuda","gloo","nccl","cuda_devices"]:
        print(f"{k}={data[k]}")


def plot(source, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(source.glob("spans-*.json"))
    events = [e for f in files for e in json.loads(f.read_text())]
    if not events: raise ValueError("No spans in source")
    origin = min(e["start_ns"] for e in events)
    colors = dict(forward="#3984b6",backward_and_sync="#7e5aa6",backward="#7385a5",gradient_average="#7e5aa6",optimizer="#51a883",
                  loss_reduce="#d09a32",injected_delay="#cb5c54")
    fig,ax=plt.subplots(figsize=(8,3.6))
    for e in events:
        ax.broken_barh([((e["start_ns"]-origin)/1e6,(e["end_ns"]-e["start_ns"])/1e6)],
                      (e["rank"]-.3,.6),facecolors=colors[e["name"]])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=n.replace("_"," ")) for n,c in colors.items() if any(e['name']==n for e in events)],
              loc="upper center",bbox_to_anchor=(.5,-.23),ncol=2,fontsize=8)
    ax.set(xlabel="Milliseconds from first recorded phase on this host",
           ylabel="Rank",yticks=sorted({e["rank"] for e in events}),
           title="Recorded host phases across workers")
    fig.tight_layout();fig.savefig(out/"timeline.png",dpi=200);plt.close(fig)
    trace = [dict(name=e["name"],cat="host phase",ph="X",pid=1,tid=e["rank"],
        ts=(e["start_ns"]-origin)/1000,dur=(e["end_ns"]-e["start_ns"])/1000,
        args=dict(step=e["step"],scope="CPU phase including waits; not GPU activity")) for e in events]
    save(out/"timeline.json",dict(traceEvents=trace))
    print(f"spans={len(events)} ranks={len(files)}")
    print("scope=host phases; durations include communication waits")
    print("saved=timeline.png timeline.json")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("case",choices=["probe","ddp","nodes","slow","checkpoint","crash","mismatch","timeout","shard","plot","gpu-ddp","gpu-shard"])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--workers",type=int,choices=[1,2,4],default=2)
    p.add_argument("--debug-level", choices=["OFF","INFO","DETAIL"],
                   help="Default INFO; mismatch defaults to DETAIL for collective checks")
    a=p.parse_args()
    debug_level=a.debug_level or ("DETAIL" if a.case=="mismatch" else "INFO")
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.case=="probe": probe(out);return
    if a.case=="plot": plot(a.source,out);return
    if a.case in ["crash","mismatch","timeout","shard","gpu-shard","nodes"] and a.workers != 2:
        raise ValueError("This exercise requires exactly two workers")
    if a.case.startswith("gpu-"):
        import torch
        import torch.distributed as dist
        if torch.cuda.device_count()<a.workers or not dist.is_nccl_available():
            save(out/"qualification.json",dict(status="UNAVAILABLE",workers=a.workers,
                cuda_devices=torch.cuda.device_count(),nccl=dist.is_nccl_available()))
            print("UNAVAILABLE: need NCCL and one distinct CUDA device per worker")
            raise SystemExit(2)
    if a.case=="nodes":
        record=capture([sys.executable,str(ROOT/"nodes.py"),"--out",str(out)],out,debug_level=debug_level)
        if record["returncode"]!=0 or record["timed_out"]:raise RuntimeError("Logical nodes failed")
        r=json.loads((out/"result.json").read_text())
        print("logical_nodes=2 workers_per_node=1 host=one loopback=True")
        print(f"training={r['status']} max_reference_error={r['max_reference_error']:.3e}")
        print("retained=node-*-stdout.txt node-*-stderr.txt rank-*.jsonl")
        return
    if a.case=="checkpoint":
        for name in ["first","resume"]:(out/name).mkdir()
        first=launch(out/"first","train",a.workers,["--steps","2"],debug_level)
        if first["returncode"]!=0 or first["timed_out"]: raise RuntimeError("Initial training failed")
        resume=launch(out/"resume","train",a.workers,["--steps","4","--resume",str(out/"first/checkpoint.pt")],debug_level)
        if resume["returncode"]!=0 or resume["timed_out"]: raise RuntimeError("Resumed training failed")
        result=json.loads((out/"resume/result.json").read_text())
        assert result["resumed_at"]==2 and result["status"]=="PASS"
        print("checkpoint=PASS next_step=2 optimizer_and_rank_rng_restored")
        print("resumed=PASS compared_with_uninterrupted_reference")
        print(f"max_parameter_error={result['max_reference_error']:.3e}")
        return
    fault=a.case in ["crash","mismatch","timeout"]
    mode="fault" if fault else "shard" if a.case in ["shard","gpu-shard"] else "train"
    args=["--fault",a.case] if fault else []
    if a.case=="slow":args += ["--delay","0.75"]
    if a.case.startswith("gpu-"):args += ["--device","cuda"]
    record=launch(out,mode,a.workers,args,debug_level)
    if fault:
        events=[json.loads(s) for f in out.glob("rank-*.jsonl") for s in f.read_text().splitlines()]
        reached={e["rank"] for e in events if e["event"]=="fault_boundary"}
        text=(out/"stderr.txt").read_text()
        injected=any(e["event"]=="injected_exit" and e["code"]==17 for e in events) if a.case=="crash" else (
            "Detected mismatch between collectives" in text if a.case=="mismatch" else
            any(e["event"]=="injected_delay" for e in events) and ("timeout" in text.lower() or "timed out" in text.lower() or "5000 ms" in text))
        ok=(a.workers==2 and reached=={0,1} and record["returncode"]!=0
            and not record["timed_out"] and injected)
        save(out/"result.json",dict(status="EXPECTED_FAILURE" if ok else "UNEXPECTED",
             case=a.case,launcher_returncode=record["returncode"],outer_timeout=record["timed_out"],
             both_ranks_reached_fault_boundary=reached=={0,1},fault_evidence_found=bool(injected)))
        print(f"case={a.case} outcome={'EXPECTED_FAILURE' if ok else 'UNEXPECTED'}")
        print(f"launcher_exit={record['returncode']} outer_timeout={record['timed_out']}")
        print("retained=stdout.txt stderr.txt rank-*.jsonl launch.json")
        if not ok:raise SystemExit(1)
    else:
        if record["returncode"]!=0 or record["timed_out"]:
            print("FAIL: inspect stdout.txt stderr.txt launch.json");raise SystemExit(1)
        # Complete worker output remains unfiltered in stdout.txt and stderr.txt.
        if mode=="train":
            r=json.loads((out/"result.json").read_text())
            print(f"training={r['status']} workers={a.workers} global_batch=8 masked_tokens=16")
            print(f"parameters={r['parameters']} dtype={r['dtype']}")
            print(f"max_reference_error={r['max_reference_error']:.3e}")
            print(f"max_replica_error={r['max_replica_error']:.3e}")
        else:
            r=json.loads((out/"shard-0.json").read_text())
            print(f"sharding=PASS local_weights={r['local_weights']} full_weights={r['full_weights']}")
            print("forward_and_both_gradients=PASS tolerance=1e-12")
        print("retained=stdout.txt stderr.txt rank-*.jsonl launch.json")


if __name__=="__main__": main()
