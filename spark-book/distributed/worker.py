"""Internal torchrun worker; launch through lab.py for bounded capture."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import make_model, batch, loss, flat, optimizer
from checkpoint_contract import BATCH_RULE, validate_checkpoint


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "fault", "shard"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--delay", type=float, default=0)
    p.add_argument("--fault", choices=["crash", "mismatch", "timeout"])
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu")
    if a.device == "cuda":
        if local >= torch.cuda.device_count() or not dist.is_nccl_available():
            raise RuntimeError("A distinct CUDA device and NCCL are required per local rank")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    dist.init_process_group("nccl" if a.device == "cuda" else "gloo",
                            timeout=timedelta(seconds=5 if a.mode == "fault" else 45))
    events = []
    def emit(name, **kw):
        record = dict(event=name, rank=rank, world=world, pid=os.getpid(),
                      monotonic_ns=time.monotonic_ns(), **kw)
        with (a.out / f"rank-{rank}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    @contextmanager
    def span(name, step):
        # CPU spans use one host monotonic clock. CUDA spans synchronize for
        # readability and therefore deliberately perturb natural overlap.
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.monotonic_ns()
        try:
            yield
        finally:
            if device.type == "cuda": torch.cuda.synchronize()
            end = time.monotonic_ns()
            events.append(dict(name=name, rank=rank, step=step,
                               start_ns=start, end_ns=end))
    emit("initialized", device=str(device), backend=dist.get_backend())
    try:
        if a.mode == "fault":
            dist.barrier()
            emit("fault_boundary", fault=a.fault)
            if a.fault == "crash" and rank == 1:
                emit("injected_exit", code=17)
                os._exit(17)
            if a.fault == "timeout" and rank == 1:
                emit("injected_delay", seconds=7)
                time.sleep(7)
            if a.fault == "mismatch" and rank == 1:
                emit("collective", operation="broadcast")
                dist.broadcast(torch.ones(1), src=0)
            else:
                emit("collective", operation="all_reduce")
                dist.all_reduce(torch.ones(1))
            raise AssertionError("Fault did not produce the expected failure")
        if a.mode == "shard":
            if world != 2: raise ValueError("Sharding exercise requires two ranks")
            # Column-parallel matrix multiplication: each worker owns half W.
            x = torch.arange(12, dtype=torch.float64, device=device).reshape(3, 4) / 8
            full_w = torch.arange(24, dtype=torch.float64, device=device).reshape(4, 6) / 16
            w = full_w[:, rank * 3:(rank + 1) * 3].clone()
            y_local = (x @ w).contiguous()
            ys = [torch.empty_like(y_local) for _ in range(world)]
            dist.all_gather(ys, y_local)
            y = torch.cat(ys, dim=1)
            dy = torch.ones_like(y_local)
            dx_part = dy @ w.T
            dist.all_reduce(dx_part)
            dw = x.T @ dy
            y_err = float((y - x @ full_w).abs().max())
            all_dy = torch.ones(3, 6, dtype=torch.float64, device=device)
            dx_err = float((dx_part - all_dy @ full_w.T).abs().max())
            dw_err = float((dw - (x.T @ all_dy)[:, rank*3:(rank+1)*3]).abs().max())
            assert max(y_err, dx_err, dw_err) < 1e-12
            write(a.out/f"shard-{rank}.json", dict(rank=rank, local_weights=w.numel(),
                full_weights=full_w.numel(), output_error=y_err,
                input_gradient_error=dx_err, weight_gradient_error=dw_err))
            if rank == 0: print("sharding=PASS forward_and_gradients tolerance=1e-12", flush=True)
            return
        if world not in (1, 2, 4): raise ValueError("Use one, two or four workers")
        dtype = torch.float64 if device.type == "cpu" else torch.float32
        model = make_model(device, dtype)
        opt = optimizer(model)
        start_step = 0
        if a.resume:
            ck = torch.load(a.resume, map_location="cpu", weights_only=True)
            validate_checkpoint(ck, world)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            start_step = ck["next_step"]
            torch.set_rng_state(ck["rng"][rank])
            if device.type == "cuda": torch.cuda.set_rng_state(ck["cuda_rng"][rank])
        ddp = DDP(model, device_ids=[local] if device.type == "cuda" else None)
        per_rank = 8 // world
        emit("data_partition", examples=list(range(rank*per_rank, (rank+1)*per_rank)),
             supervised_tokens=2*per_rank)
        losses = []
        dist.barrier()
        for step in range(start_step, a.steps):
            if rank == 1 and a.delay and step == 1:
                with span("injected_delay", step): time.sleep(a.delay)
            ids, labels = batch(step, device)
            ids = ids[rank*per_rank:(rank+1)*per_rank]
            labels = labels[rank*per_rank:(rank+1)*per_rank]
            opt.zero_grad(set_to_none=True)
            with span("forward", step): value = loss(ddp, ids, labels)
            with span("backward_and_sync", step): value.backward()
            with span("optimizer", step): opt.step()
            mean = value.detach().clone()
            with span("loss_reduce", step):
                dist.all_reduce(mean)
                mean /= world
            losses.append(float(mean))
            emit("step_complete", step=step, loss=float(mean))
            if rank == 0: print(f"step={step} global_loss={float(mean):.9f}", flush=True)
        actual = flat(model)
        leader = actual.clone()
        dist.broadcast(leader, src=0)
        replica_error = (actual-leader).abs().max()
        dist.all_reduce(replica_error, op=dist.ReduceOp.MAX)
        rng = [None] * world
        dist.all_gather_object(rng, torch.get_rng_state())
        cuda_rng = [None] * world
        if device.type == "cuda": dist.all_gather_object(cuda_rng, torch.cuda.get_rng_state().cpu())
        if rank == 0:
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                next_step=a.steps, world=world, rng=rng, cuda_rng=cuda_rng,
                batch_rule=BATCH_RULE), a.out/"checkpoint.pt")
            reference = make_model(device, dtype)
            ropt = optimizer(reference)
            for step in range(a.steps):
                ropt.zero_grad(set_to_none=True)
                ids, labels = batch(step, device)
                loss(reference, ids, labels).backward()
                ropt.step()
            error = float((actual-flat(reference)).abs().max())
            tolerance = 1e-9 if device.type == "cpu" else 1e-4
            passed = error <= tolerance and float(replica_error) <= tolerance
            result = dict(status="PASS" if passed else "FAIL", workers=world,
                device=str(device), steps=a.steps, resumed_at=start_step,
                global_batch=8, masked_tokens=16, parameters=actual.numel(),
                dtype=str(dtype), max_reference_error=error,
                max_replica_error=float(replica_error), tolerance=tolerance,
                losses=losses, scope="training correctness; not a scaling benchmark")
            write(a.out/"result.json", result)
            print(f"reference={'PASS' if passed else 'FAIL'} tolerance={tolerance:g}", flush=True)
            print(f"max_parameter_error={error:.3e}", flush=True)
            if not passed: raise AssertionError("Training differs from reference")
        dist.barrier()
    except Exception as e:
        emit("exception", type=type(e).__name__, message=str(e))
        raise
    finally:
        write(a.out/f"spans-{rank}.json", events)
        dist.destroy_process_group()


if __name__ == "__main__": main()
