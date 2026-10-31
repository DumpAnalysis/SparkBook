"""Validate the serialized graph using ONNX's CPU reference evaluator."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from infer import encode,tokenize

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=onnx.load(Path(a.repository)/'lab_embedding_gpu/1/model.onnx');onnx.checker.check_model(model)
    x=np.array([tokenize('alpha'),tokenize('beta')],dtype=np.int64)
    actual=ReferenceEvaluator(model).run(None,{'TOKENS':x})[0]
    error=float(np.max(np.abs(actual-encode(x))));assert actual.shape==(2,64) and error<1e-6
    (out/'result.json').write_text(json.dumps(dict(status='PASS',onnx=onnx.__version__,max_abs_error=error,scope='CPU ONNX reference evaluator; no ONNX Runtime or Triton'),indent=2)+'\n')
    print(f'PASS ONNX {onnx.__version__} checker and CPU reference evaluator')
    print(f'opset=18 IR=9 shape=2x64 max_abs_error={error:.3e}')
    print('No ONNX Runtime, Triton or GPU execution')
if __name__=='__main__':main()
