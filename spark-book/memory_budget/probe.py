#!/usr/bin/env python3
"""Exercise 251: read-only snapshots, or bounded CUDA allocation qualification.
No CPU fallback, automatic installation, driver change, or allocation escalation.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

GIB = 1024**3
MIB = 1024**2
MAX_GIB = 18
MIN_GUEST_AVAILABLE = 2 * GIB
MIN_CUDA_RESERVE = 4 * GIB
CHECK_TILE = 8 * MIB


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps(dict(utc=utc(), event=event, **fields), sort_keys=True), flush=True)


def guest_info():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            result[key + '_bytes'] = int(value.split()[0]) * 1024
    return result


def capture(argv):
    start = utc()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return dict(argv=argv, started_utc=start, ended_utc=utc(),
                    exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=argv, started_utc=start, ended_utc=utc(),
                    status='UNAVAILABLE', error=str(exc))


def snapshot(torch, stage):
    free, total = torch.cuda.mem_get_info(0)
    data = dict(stage=stage, cuda_free_bytes=free, cuda_total_bytes=total,
                torch_allocated_bytes=torch.cuda.memory_allocated(0),
                torch_reserved_bytes=torch.cuda.memory_reserved(0),
                guest=guest_info())
    emit('memory', **data)
    return data


def require_headroom(state, next_bytes):
    if state['guest']['MemAvailable_bytes'] < MIN_GUEST_AVAILABLE:
        raise RuntimeError('GUARD: guest MemAvailable is below 2 GiB')
    if state['cuda_free_bytes'] < next_bytes + MIN_CUDA_RESERVE:
        raise RuntimeError('GUARD: next block would breach 4 GiB reported CUDA reserve')


def verify_blocks(torch, blocks):
    """Read every retained byte in bounded tiles; no full-size CPU copy."""
    checked = 0
    for block_number, tensor in enumerate(blocks):
        expected = (block_number % 251) + 1
        for start in range(0, tensor.numel(), CHECK_TILE):
            tile = tensor[start:start + CHECK_TILE]
            if not bool(torch.all(tile == expected).item()):
                raise RuntimeError(f'CORRUPTION: block {block_number}, tile {start}')
            checked += tile.numel()
    torch.cuda.synchronize(0)
    return checked


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocate-gib', type=int, default=0,
                   help='0 queries only; integer 1..18 retains this many 1 GiB blocks')
    p.add_argument('--host-note', default='',
                   help='Manually observed Windows Available RAM, time and other workload state')
    args = p.parse_args(argv)
    if not 0 <= args.allocate_gib <= MAX_GIB:
        p.error('--allocate-gib must be an integer from 0 through 18')
    if args.allocate_gib and not args.host_note.strip():
        p.error('--host-note is required for allocation; record actual host observations')
    return args


def main(argv=None):
    args = parse(argv)
    emit('identity', interpreter=sys.executable, python=sys.version,
         platform=platform.platform(), machine=platform.machine(), pid=os.getpid(),
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         allocation_target_bytes=args.allocate_gib * GIB,
         host_note=args.host_note, execution_scope='Actual executing environment only')
    emit('free_h_before', capture=capture(['free', '-h']))
    smi = shutil.which('nvidia-smi')
    if not smi and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        smi = '/usr/lib/wsl/lib/nvidia-smi'
    emit('nvidia_smi_before', capture=capture([smi]) if smi else {'status': 'UNAVAILABLE'})
    try:
        import torch
    except ImportError as exc:
        emit('result', status='UNAVAILABLE', reason=str(exc))
        return 3
    emit('framework', torch=torch.__version__, framework_cuda=torch.version.cuda,
         cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        emit('result', status='UNAVAILABLE', reason='CUDA is unavailable; no CPU fallback')
        return 3
    blocks = []
    tensor = None
    code = 1
    try:
        torch.cuda.set_device(0)
        emit('device', name=torch.cuda.get_device_name(0),
             capability=list(torch.cuda.get_device_capability(0)))
        before = snapshot(torch, 'before')
        if not args.allocate_gib:
            emit('result', status='OBSERVED', allocation_executed=False,
                 note='Queries only; no capacity qualification')
            code = 0
        else:
            if 'microsoft' not in platform.release().lower():
                raise RuntimeError('GUARD: allocation exercise requires a WSL2 kernel')
            start = time.perf_counter()
            for number in range(args.allocate_gib):
                state = snapshot(torch, f'before_block_{number}')
                require_headroom(state, GIB)
                tensor = torch.empty(GIB, dtype=torch.uint8, device='cuda:0')
                blocks.append(tensor)
                tensor.fill_((number % 251) + 1)
                tensor = None
                torch.cuda.synchronize(0)
                emit('written', retained_bytes=len(blocks) * GIB,
                     blocks=len(blocks))
            at_target = snapshot(torch, 'target_written')
            require_headroom(at_target, 0)
            checked = verify_blocks(torch, blocks)
            assert checked == args.allocate_gib * GIB
            emit('result', status='PASSED', allocated_written_verified_bytes=checked,
                 retained_blocks=len(blocks), guest_MemTotal_bytes=before['guest']['MemTotal_bytes'],
                 exceeds_guest_MemTotal=checked > before['guest']['MemTotal_bytes'],
                 operation_seconds=time.perf_counter() - start,
                 limits='Not maximum capacity, residency, model fit, bandwidth or training qualification')
            code = 0
    except KeyboardInterrupt:
        emit('result', status='INTERRUPTED', retained_blocks=len(blocks))
        code = 130
    except Exception as exc:
        emit('result', status='FAILED', error_type=type(exc).__name__,
             error=str(exc), retained_blocks=len(blocks))
        code = 1
    finally:
        tensor = None
        blocks.clear()
        try:
            torch.cuda.empty_cache()
            snapshot(torch, 'after_release')
        except Exception as exc:
            emit('release_observation_failed', error_type=type(exc).__name__, error=str(exc))
            if code == 0:
                code = 1
        emit('free_h_after', capture=capture(['free', '-h']))
        emit('nvidia_smi_after', capture=capture([smi]) if smi else {'status': 'UNAVAILABLE'})
        emit('process_result', exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
