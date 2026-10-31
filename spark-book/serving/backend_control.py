"""Test repository logic with a small API double, not a running Triton server."""
import argparse,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
from infer import encode,tokenize
class Tensor:
    def __init__(self,name,array):self.name=name;self.array=array
    def as_numpy(self):return self.array
class Response:
    def __init__(self,output_tensors=None,error=None):self.output_tensors=output_tensors;self.error=error

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    fake=types.ModuleType('triton_python_backend_utils')
    fake.Tensor=Tensor;fake.InferenceResponse=Response;fake.TritonError=ValueError
    fake.get_input_tensor_by_name=lambda request,name:request[name]
    sys.modules[fake.__name__]=fake
    path=Path(a.repository)/'lab_embedding/1/model.py'
    spec=importlib.util.spec_from_file_location('local_backend',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=module.TritonPythonModel();m.initialize({})
    x=np.array([tokenize('alpha'),tokenize('beta'),tokenize('gamma')],dtype=np.int64)
    result=m.execute([{'TOKENS':Tensor('TOKENS',x[:1])},{'TOKENS':Tensor('TOKENS',x[1:])}])
    assert [r.output_tensors[0].as_numpy().shape for r in result]==[(1,64),(2,64)]
    actual=np.concatenate([r.output_tensors[0].as_numpy() for r in result])
    error=float(np.max(np.abs(actual-encode(x))));assert error<1e-6
    bad=m.execute([{'TOKENS':Tensor('TOKENS',np.zeros((1,15),dtype=np.int64))}]);assert bad[0].error is not None
    record=dict(status='PASS',request_batch_sizes=[1,2],max_abs_error=error,invalid_shape='rejected',scope='Python backend API double; not Triton execution')
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS Python backend combined requests=2 batches=1+2 outputs=1+2')
    print(f'max_abs_error={error:.3e} invalid_shape=rejected')
    print('API double only; no Triton scheduler, HTTP or GPU tested')
if __name__=='__main__':main()
