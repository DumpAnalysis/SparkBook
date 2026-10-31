"""Small offline controls. None of these commands execute a GPU framework."""
import argparse, csv, hashlib, importlib.metadata as md, importlib.util
import json, platform, shutil, sys
from pathlib import Path
import numpy as np

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixture(out):
    rng=np.random.default_rng(1401)
    w=rng.normal(0,.3,(8,4)).astype('float32')
    w[:,3]*=.02  # a small channel exposes the per-tensor tradeoff
    b=np.array([.1,-.2,.3,.01],dtype='float32')
    cal=rng.uniform(-1,1,(96,8)).astype('float32')
    test=rng.uniform(-1,1,(32,8)).astype('float32')
    shifted=test.copy(); shifted[0,0]=12
    z=rng.normal(size=(64,2))*[4,2]
    basis=np.linalg.qr(rng.normal(size=(16,2)))[0]
    vectors=(z@basis.T+rng.normal(0,.01,(64,16))).astype('float32')
    np.savez(out/'fixture.npz',w=w,b=b,cal=cal,test=test,shifted=shifted,vectors=vectors)
    rows=[dict(id=i,route=['embedding','generation'][i%2],ok=int(i%7!=0),
               latency_ms=10+(i%5)*3,bytes=100+i) for i in range(28)]
    with (out/'requests.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    dump(out/'identity.json',{'fixture_sha256':sha(out/'fixture.npz'),'rows':len(rows),
         'provenance':'synthetic controls; not measured laptop traffic'})
    print('Created fixed fixture: 28 requests, 64 vectors, 96 calibration and 32 held-out rows')
    print('Input X float32[N,8]; output Y float32[N,4]; N is dynamic')

def analytics(src,out):
    rows=list(csv.DictReader((src/'requests.csv').open()))
    result=[]
    for route in sorted({r['route'] for r in rows}):
        good=[r for r in rows if r['route']==route and int(r['ok'])==1]
        result.append(dict(route=route,count=len(good),mean_ms=float(np.mean([int(r['latency_ms']) for r in good]))))
    x=np.load(src/'fixture.npz')['vectors']; centered=x-x.mean(0)
    _,s,v=np.linalg.svd(centered,full_matrices=False)
    projection=v[:2].T@v[:2];scores=centered@v[:2].T
    ratio=float(np.sum(s[:2]**2)/np.sum(s**2))
    np.savez(out/'pca.npz',projection=projection,scores=scores)
    dump(out/'analytics.json',dict(groups=result,rows=len(rows),accepted=sum(r['count'] for r in result),variance_ratio=ratio))
    for r in result:print(f"{r['route']}: accepted={r['count']} mean_ms={r['mean_ms']:.6f}")
    print(f'CPU PCA: shape={scores.shape} explained_ratio={ratio:.6f}')
    print('Compare PCA projectors, not signed component coordinates')

def make_onnx(src,out):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    a=np.load(src/'fixture.npz')
    nodes=[h.make_node('MatMul',['X','W'],['M'],name='projection'),h.make_node('Add',['M','B'],['A'],name='bias'),h.make_node('Relu',['A'],['Y'],name='activation')]
    g=h.make_graph(nodes,'diagnostic-linear',[h.make_tensor_value_info('X',T.FLOAT,['N',8])],
                   [h.make_tensor_value_info('Y',T.FLOAT,['N',4])],
                   [nh.from_array(a['w'],'W'),nh.from_array(a['b'],'B')])
    m=h.make_model(g,opset_imports=[h.make_opsetid('',18)],ir_version=9)
    onnx.checker.check_model(m);m=onnx.shape_inference.infer_shapes(m)
    onnx.save(m,out/'linear.onnx')
    dump(out/'onnx.json',dict(onnx=onnx.__version__,opset=18,ir=9,sha256=sha(out/'linear.onnx'),nodes=3))
    print('ONNX checker and shape inference passed; nodes=3 opset=18 IR=9')
    print('No GPU execution is implied by a valid ONNX file')

def oracle(a,x):return np.maximum(x@a['w']+a['b'],0)

def ort_run(src,models,out,provider='CPUExecutionProvider'):
    import onnxruntime as ort
    if provider not in ort.get_available_providers():
        raise RuntimeError(f'requested provider unavailable: {provider}; available={ort.get_available_providers()}')
    so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
    so.enable_profiling=True;so.profile_file_prefix=str(out/'ort')
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath=str(out/'optimized.onnx')
    # Fail creation if unsupported nodes would be assigned to CPU in the GPU lab.
    if provider!='CPUExecutionProvider':so.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    session=ort.InferenceSession(str(models/'linear.onnx'),so,providers=[(provider, {'use_tf32':'0'})] if provider=='CUDAExecutionProvider' else [provider])
    session.disable_fallback()
    if provider not in session.get_providers():raise RuntimeError('provider was not registered in the session')
    a=np.load(src/'fixture.npz');errors=[]
    for n in (1,3,8):
        x=a['test'][:n];y=session.run(None,{'X':x})[0]
        error=float(np.max(np.abs(y-oracle(a,x))));errors.append(error)
        np.savez(out/f'batch-{n}.npz',x=x,y=y)
        print('measured error',error,'tolerance',1e-5,flush=True)
        assert y.shape==(n,4) and error<1e-5
        print(f'batch={n} shape={y.shape} max_abs_error={error:.3e}')
    failures=[]
    for label,x in [('width',np.zeros((2,7),dtype='float32')),('dtype',np.zeros((2,8),dtype='float64'))]:
        try:session.run(None,{'X':x})
        except Exception as e:
            failures.append(label);(out/f'{label}-error.txt').write_text(str(e)+'\n')
            print(f'invalid {label}: rejected ({type(e).__name__})')
        else:raise AssertionError('invalid input accepted')
    profile=Path(session.end_profiling());events=json.loads(profile.read_text())
    observed=sorted({e.get('args',{}).get('provider') for e in events if e.get('args',{}).get('provider')})
    assert provider in observed,(provider,observed)
    if provider!='CPUExecutionProvider':assert 'CPUExecutionProvider' not in observed
    report=dict(version=ort.__version__,requested=provider,registered=session.get_providers(),observed=observed,
                max_error=max(errors),rejections=failures,profile=profile.name,provider_options=session.get_provider_options(),build_info=ort.get_build_info())
    dump(out/'ort.json',report)
    print('Observed node providers: '+', '.join(observed))

def qdq(x,scale):return np.clip(np.rint(x/scale),-128,127).astype('int8').astype('float32')*scale

def quantize(src,out):
    import onnx, onnxruntime as ort
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    a=np.load(src/'fixture.npz');w=a['w'];b=a['b']
    sx=np.float32(np.max(np.abs(a['cal']))/127)
    sw=np.maximum(np.max(np.abs(w),axis=0)/127,np.finfo('float32').tiny).astype('float32')
    wt=qdq(w,np.float32(np.max(np.abs(w))/127));wc=qdq(w,sw)
    zero=np.array(0,dtype='int8');zw=np.zeros(4,dtype='int8')
    # Keep W in float32 so Q->DQ is explicit and the teaching graph is inspectable.
    inits=[nh.from_array(v,k) for k,v in [('W',w),('B',b),('sx',sx),('sw',sw),('zx',zero),('zw',zw)]]
    nodes=[h.make_node('QuantizeLinear',['X','sx','zx'],['Xq']),h.make_node('DequantizeLinear',['Xq','sx','zx'],['Xd']),
           h.make_node('QuantizeLinear',['W','sw','zw'],['Wq'],axis=1),h.make_node('DequantizeLinear',['Wq','sw','zw'],['Wd'],axis=1),
           h.make_node('MatMul',['Xd','Wd'],['M']),h.make_node('Add',['M','B'],['A']),h.make_node('Relu',['A'],['Y'])]
    m=h.make_model(h.make_graph(nodes,'explicit-qdq',[h.make_tensor_value_info('X',T.FLOAT,['N',8])],
          [h.make_tensor_value_info('Y',T.FLOAT,['N',4])],inits),opset_imports=[h.make_opsetid('',18)],ir_version=9)
    onnx.checker.check_model(m);onnx.save(m,out/'linear-qdq.onnx')
    so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(str(out/'linear-qdq.onnx'),so,providers=['CPUExecutionProvider'])
    metrics={}
    for name in ['test','shifted']:
        x=a[name];ref=oracle(a,x);y=np.maximum(qdq(x,sx)@wc+b,0)
        actual=session.run(None,{'X':x})[0]
        assert np.max(np.abs(actual-y))<1e-5
        metrics[name]=dict(mae=float(np.mean(np.abs(y-ref))),max_error=float(np.max(np.abs(y-ref))),
             clipped=int(np.count_nonzero(np.abs(x)>sx*127)),ort_vs_numpy=float(np.max(np.abs(actual-y))))
        print(f"{name}: clipped={metrics[name]['clipped']} MAE={metrics[name]['mae']:.6f} max_error={metrics[name]['max_error']:.6f}")
    tiny_tensor=float(np.mean(np.abs(w[:,3]-wt[:,3])));tiny_channel=float(np.mean(np.abs(w[:,3]-wc[:,3])))
    print(f'small-channel weight MAE: per-tensor={tiny_tensor:.7f} per-channel={tiny_channel:.7f}')
    print(f'INT8 weight payload={w.size} bytes; scales={sw.nbytes} bytes; FP32 weights={w.nbytes} bytes')
    print('QDQ teaching graph retains FP32 W: graph size is not compressed payload size')
    dump(out/'quantization.json',dict(metrics=metrics,activation_scale=float(sx),weight_scales=sw.tolist(),
         calibration_rows=96,test_rows=32,calibration_sha256=hashlib.sha256(a['cal'].tobytes()).hexdigest(),
         per_tensor_small_mae=tiny_tensor,per_channel_small_mae=tiny_channel,
         float_weight_bytes=w.nbytes,int8_payload_bytes=w.size,scale_bytes=sw.nbytes,
         qdq_file_bytes=(out/'linear-qdq.onnx').stat().st_size))

def data_audit(out):
    # Byte token IDs make the control independent of an external tokenizer.
    rows=[dict(id=f'case-{i}',prompt=f'Diagnose case {i}: ',answer=f'collect trace {i}') for i in range(12)]
    splits={'train':rows[:8],'validation':rows[8:10],'test':rows[10:]}
    fingerprints={k:{hashlib.sha256((r['prompt']+r['answer']).encode()).hexdigest() for r in v} for k,v in splits.items()}
    assert not fingerprints['train'] & (fingerprints['validation']|fingerprints['test'])
    for k,v in splits.items():
        (out/f'{k}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in v))
    row=rows[0];p=list(row['prompt'].encode());ans=list(row['answer'].encode())+[256]
    ids=p+ans;labels=[-100]*len(p)+ans
    # Causal training consumes logits[:-1] and labels[1:].
    supervised=sum(v!=-100 for v in labels[1:]);assert supervised==len(ans)
    contaminated=fingerprints['validation']|{next(iter(fingerprints['train']))}
    duplicates=len(contaminated&fingerprints['train']);assert duplicates==1
    dump(out/'audit.json',dict(split_counts={k:len(v) for k,v in splits.items()},overlap=0,
          injected_overlap_detected=duplicates,prompt_tokens=len(p),answer_including_eos=len(ans),
          shifted_supervised_tokens=supervised,input_ids=ids,labels=labels,
          tokenizer='UTF-8 byte control with EOS 256; NOT a NeMo or Qwen tokenizer'))
    print('split rows: train=8 validation=2 test=2; exact content overlap=0')
    print('Injected train/validation duplicate detected=1')
    print(f'prompt tokens={len(p)}; shifted answer labels including EOS={supervised}')
    print('Production audit must use the real tokenizer and rendered chat template')

def lora_control(out):
    import torch
    torch.manual_seed(1402);torch.set_num_threads(1)
    base=torch.randn(4,8);saved=base.clone()
    a=torch.nn.Parameter(torch.randn(2,8)*.1);b=torch.nn.Parameter(torch.zeros(4,2))
    x=torch.randn(16,8);target=x@(base+.2).T
    optimizer=torch.optim.SGD([a,b],lr=.02)
    def loss():return ((x@(base+b@a).T-target)**2).mean()
    start=loss().item()
    for _ in range(20):
        optimizer.zero_grad();loss().backward();optimizer.step()
    final=loss().item();assert final<start and torch.equal(base,saved)
    np.savez(out/'adapter.npz',a=a.detach().numpy(),b=b.detach().numpy(),base=base.numpy(),x=x.numpy(),target=target.numpy())
    ck=np.load(out/'adapter.npz');restored=ck['x']@(ck['base']+ck['b']@ck['a']).T
    err=float(np.max(np.abs(restored-(x@(base+b@a).T).detach().numpy())))
    assert err<1e-5
    dump(out/'lora.json',dict(torch=torch.__version__,steps=20,initial_loss=start,final_loss=final,
      frozen_base=True,restore_max_error=err,trainable_parameters=24,base_parameters=32,
      execution='tiny CPU algebra control; not NeMo training'))
    print(f'CPU LoRA control: steps=20 trainable=24 base=32 frozen=True')
    print(f'loss {start:.6f} -> {final:.6f}; restored max_error={err:.3e}')
    print('No NeMo, language-model quality or GPU training result is claimed')

def inventory(out):
    packages={}
    for name in ['numpy','onnx','onnxruntime','torch','cudf','cuml','rmm','tensorrt','nemo_automodel']:
        packages[name]=importlib.util.find_spec(name) is not None
    report=dict(machine=platform.machine(),python=platform.python_version(),modules=packages,
        executables={k:shutil.which(k) is not None for k in ['nvidia-smi','nvcc','docker','trtexec','automodel']})
    dump(out/'inventory.json',report)
    print(f"Preparation machine={report['machine']} Python={report['python']}")
    print('Modules present: '+', '.join(k for k,v in packages.items() if v))
    print('Modules absent: '+', '.join(k for k,v in packages.items() if not v))
    print('Target GPU frameworks: NOT EXECUTED')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['fixture','analytics','onnx','ort','quantize','data','lora','inventory'])
    ap.add_argument('--src',type=Path);ap.add_argument('--models',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--provider',default='CPUExecutionProvider');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.command in ['fixture','data','lora','inventory']:
        {'fixture':fixture,'data':data_audit,'lora':lora_control,'inventory':inventory}[args.command](args.out)
    elif args.command=='ort':ort_run(args.src,args.models,args.out,args.provider)
    else:{'analytics':analytics,'onnx':make_onnx,'quantize':quantize}[args.command](args.src,args.out)

if __name__=='__main__':main()
