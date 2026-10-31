"""CUDA-only two-stream correctness and activity-capture qualification."""
import argparse
from pathlib import Path
import sys
from core import write


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        write(out/'result.json',dict(status='UNAVAILABLE',reason='CUDA unavailable'))
        print('UNAVAILABLE CUDA unavailable; no stream run executed');return 2
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(710)
    # Host buffers remain alive until all streams have completed.
    host=[torch.randn(256,256).pin_memory() for _ in range(4)]
    ref=[host[0]@host[1],host[2]@host[3]]
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    result=[];pairs=[]
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    # Warm up the default path outside the capture.
    z=host[0].to('cuda');_ = z@z;torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
        for i,s in enumerate(streams):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s),torch.profiler.record_function(f'stream_job_{i}'):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                x=host[2*i].to('cuda',non_blocking=True)
                y=host[2*i+1].to('cuda',non_blocking=True)
                v=x@y;end.record()
                # Keep device inputs alive and declare their stream usage.
                x.record_stream(s);y.record_stream(s)
                result.append(v);pairs.append((start,end,x,y))
        for s in streams:torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out/'streams-profiler.json'))
    errors=[float((v.cpu()-r).abs().max()) for v,r in zip(result,ref)]
    for v,r in zip(result,ref):torch.testing.assert_close(v.cpu(),r,atol=1e-4,rtol=1e-4)
    durations=[a.elapsed_time(b) for a,b,_,_ in pairs]
    write(out/'result.json',dict(status='PASS',device=torch.cuda.get_device_name(0),
          max_abs_errors=errors,stream_elapsed_ms=durations,
          overlap='MUST_INSPECT_DEVICE_TRACE',torch=torch.__version__,
          cuda_runtime=torch.version.cuda))
    print('PASS two stream outputs match CPU reference')
    print('stream_elapsed_ms='+','.join(f'{x:.3f}' for x in durations))
    print('overlap=UNDETERMINED inspect streams-profiler.json');return 0


if __name__=='__main__':sys.exit(main())
