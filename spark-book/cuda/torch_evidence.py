"""Bounded allocator snapshots and operator traces in a qualified environment."""
import argparse,json,pathlib,struct

def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['memory','trace']);p.add_argument('--out',required=True)
    args=p.parse_args();out=pathlib.Path(args.out)
    if out.exists():raise SystemExit('Use a new output directory')
    if not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
    torch.cuda.init();out.mkdir(parents=True)
    (out/'versions.json').write_text(json.dumps({'torch':torch.__version__,
        'cuda_build':torch.version.cuda,'mode':args.mode},indent=2))
    torch.manual_seed(42)
    if args.mode=='memory':
        # Underscored APIs are version-sensitive; recorded host framework is torch 2.14.1+cu130.
        torch.cuda.memory._record_memory_history(max_entries=2000)
        rows=[]
        def record(label):
            torch.cuda.synchronize()
            rows.append({'stage':label,'allocated':torch.cuda.memory_allocated(),
                         'reserved':torch.cuda.memory_reserved()})
        try:
            x=torch.arange(64,device='cuda',dtype=torch.float32)
            a=torch.ones((1024,1024),device='cuda');b=a*2
            record('both_live')
            torch.cuda.memory._dump_snapshot(str(out/'live.pickle'))
            values=x.detach().cpu().contiguous().tolist()
            (out/'tensor.f32le').write_bytes(struct.pack('<64f',*values))
            del b;record('b_deleted')
            torch.cuda.memory._dump_snapshot(str(out/'after_free.pickle'))
            torch.cuda.empty_cache();record('cache_released')
            (out/'memory.json').write_text(json.dumps(rows,indent=2))
            print('tensor_values=64 tensor_bytes=256')
            print('saved=live.pickle,after_free.pickle,memory.json,tensor.f32le')
        finally:torch.cuda.memory._record_memory_history(enabled=None)
    else:
        from torch.profiler import profile,ProfilerActivity,record_function
        activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]
        if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise SystemExit('This build cannot collect CUDA profiler activity')
        x=torch.randn(512,512,device='cuda');y=x@x;torch.cuda.synchronize()
        with profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=True) as prof:
            with record_function('diagnostic_iteration'):
                for _ in range(3):y=torch.relu(x@x)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/'trace.json'))
        (out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=-1))
        print('saved=trace.json,operators.txt')
        print('Check that the trace contains GPU kernels, not only CPU events.')
    print('directory='+str(out))
if __name__=='__main__':main()
