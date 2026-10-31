"""TARGET ONLY: TensorRT 11 engine build, named tensor execution and value checks."""
import argparse,json,platform,time
from pathlib import Path
import numpy as np
from lab import dump,sha,oracle,qdq

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--src',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qdq',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import torch,tensorrt as trt
    if int(trt.__version__.split('.')[0])!=11:raise RuntimeError('This procedure targets TensorRT 11.x')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable to this PyTorch build')
    logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
    network=builder.create_network();parser=trt.OnnxParser(network,logger)
    if not parser.parse(args.model.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256*1024**2)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    profile=builder.create_optimization_profile();profile.set_shape('X',(1,8),(4,8),(8,8))
    assert [tuple(x) for x in profile.get_shape('X')]==[(1,8),(4,8),(8,8)]
    config.add_optimization_profile(profile)
    free_start,total=torch.cuda.mem_get_info();start=time.perf_counter()
    blob=builder.build_serialized_network(network,config)
    if blob is None:raise RuntimeError('engine build failed')
    build_seconds=time.perf_counter()-start;(args.out/'linear.plan').write_bytes(bytes(blob))
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(blob)
    if engine is None:raise RuntimeError('engine deserialization failed')
    inspector=engine.create_engine_inspector()
    (args.out/'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    assert engine.num_io_tensors==2
    for name in ['X','Y']:assert engine.get_tensor_dtype(name)==trt.float32
    context=engine.create_execution_context();stream=torch.cuda.Stream();fixture=np.load(args.src/'fixture.npz')
    errors=[];samples=[]
    for n in [1,4,8]:
        assert context.set_input_shape('X',(n,8))
        shape=tuple(context.get_tensor_shape('Y'));assert shape==(n,4)
        x=torch.from_numpy(fixture['test'][:n].copy()).cuda();y=torch.empty(shape,device='cuda',dtype=torch.float32)
        torch.cuda.synchronize()  # ensure the default-stream copy completed
        assert context.set_tensor_address('X',x.data_ptr())
        assert context.set_tensor_address('Y',y.data_ptr())
        for _ in range(5):assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        for _ in range(20):assert context.execute_async_v3(stream.cuda_stream)
        end_event.record(stream);stream.synchronize();device_ms=start_event.elapsed_time(end_event)/20
        result=y.cpu().numpy();ref=oracle(fixture,fixture['test'][:n])
        if args.qdq:
            sx=np.float32(np.max(np.abs(fixture['cal']))/127)
            sw=(np.max(np.abs(fixture['w']),axis=0)/127).astype('float32')
            ref=np.maximum(qdq(fixture['test'][:n],sx)@qdq(fixture['w'],sw)+fixture['b'],0)
        error=float(np.max(np.abs(result-ref)));errors.append(error)
        np.savez(args.out/f'batch-{n}.npz',x=fixture['test'][:n],y=result,reference=ref)
        assert error<1e-3,(n,error)
        samples.append(dict(batch=n,max_error=error,device_ms=device_ms,repetitions=20,warmups=5))
        print(f'batch={n} max_abs_error={error:.6g} device_ms={device_ms:.6f}')
    rejected=not context.set_input_shape('X',(9,8))
    if not rejected:raise RuntimeError('out-of-profile batch unexpectedly accepted')
    free_end,_=torch.cuda.mem_get_info()
    dump(args.out/'tensorrt.json',dict(tensorrt=trt.__version__,torch=torch.__version__,torch_cuda=torch.version.cuda,
       cpu_arch=platform.machine(),gpu=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability()),
       model_sha256=sha(args.model),plan_sha256=sha(args.out/'linear.plan'),build_seconds=build_seconds,
       workspace_limit_bytes=256*1024**2,driver_total=total,driver_free_before=free_start,driver_free_after=free_end,
       samples=samples,out_of_profile_rejected=rejected,comparison='QDQ numerical oracle' if args.qdq else 'FP32 NumPy oracle'))
    print('batch=9 rejected by optimization profile; engine and complete outputs retained')

if __name__=='__main__':main()
