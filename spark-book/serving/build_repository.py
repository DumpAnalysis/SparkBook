"""Build a versioned Triton repository; ONNX creation is an optional extra."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from infer import E,W,B,REVISION

def build_onnx(path):
    import onnx
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    init=[nh.from_array(v,n) for n,v in [('E',E),('WT',W.T.copy()),('B',B),
            ('axes',np.array([1],dtype=np.int64)),('eps',np.array(1e-12,dtype=np.float32))]]
    nodes=[h.make_node('Gather',['E','TOKENS'],['lookup'],axis=0),
           h.make_node('ReduceMean',['lookup','axes'],['mean'],keepdims=0),
           h.make_node('MatMul',['mean','WT'],['projected']),h.make_node('Add',['projected','B'],['biased']),
           h.make_node('Tanh',['biased'],['y']),h.make_node('Mul',['y','y'],['square']),
           h.make_node('ReduceSum',['square','axes'],['sum'],keepdims=1),h.make_node('Sqrt',['sum'],['norm']),
           h.make_node('Max',['norm','eps'],['safe']),h.make_node('Div',['y','safe'],['EMBEDDINGS'])]
    graph=h.make_graph(nodes,'lab_embedding',[h.make_tensor_value_info('TOKENS',T.INT64,['N',16])],
                       [h.make_tensor_value_info('EMBEDDINGS',T.FLOAT,['N',64])],init)
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',18)],producer_name='spark-book-v13');model.ir_version=9
    onnx.checker.check_model(model);onnx.save(model,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--onnx',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    name='lab_embedding_gpu' if a.onnx else 'lab_embedding';model=out/name;(model/'1').mkdir(parents=True)
    if a.onnx:build_onnx(model/'1/model.onnx')
    else:
        shutil.copyfile(Path(__file__).with_name('triton_model.py'),model/'1/model.py')
        np.savez(model/'1/weights.npz',embedding=E,projection=W,bias=B)
    backend='onnxruntime' if a.onnx else 'python';kind='KIND_GPU' if a.onnx else 'KIND_CPU'
    config=f'''name: "{name}"
backend: "{backend}"
max_batch_size: 8
input [ {{ name: "TOKENS" data_type: TYPE_INT64 dims: [16] }} ]
output [ {{ name: "EMBEDDINGS" data_type: TYPE_FP32 dims: [64] }} ]
instance_group [ {{ count: 1 kind: {kind} }} ]
version_policy: {{ specific: {{ versions: [1] }} }}
dynamic_batching {{
  max_queue_delay_microseconds: 2000
  default_queue_policy {{ max_queue_size: 16 timeout_action: REJECT default_timeout_microseconds: 1000000 }}
}}
'''
    (model/'config.pbtxt').write_text(config)
    (out/'model-identity.json').write_text(json.dumps({'revision':REVISION,'backend':backend,'instance_kind':kind},indent=2))
    print(f'GENERATED version=1 backend={backend} instance={kind} max_batch=8 queue_limit=16')
    print('Repository files written; no Triton server execution implied')

if __name__=='__main__':main()
