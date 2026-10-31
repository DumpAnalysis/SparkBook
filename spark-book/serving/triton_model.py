"""Copy into <repository>/lab_embedding/1/model.py. CPU NumPy backend."""
import json
from pathlib import Path
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self,args):
        data=np.load(Path(__file__).with_name('weights.npz'),allow_pickle=False)
        self.e,self.w,self.b=data['embedding'],data['projection'],data['bias']
    def execute(self,requests):
        arrays=[]
        for r in requests:
            x=pb_utils.get_input_tensor_by_name(r,'TOKENS').as_numpy()
            if x.dtype!=np.int64 or x.ndim!=2 or x.shape[1]!=16 or np.any(x<0) or np.any(x>255):
                return [pb_utils.InferenceResponse(error=pb_utils.TritonError('invalid TOKENS')) for _ in requests]
            arrays.append(x)
        # Python backend passes a list of requests: concatenate explicitly,
        # perform one vectorized call, then restore each request's batch.
        x=np.concatenate(arrays,axis=0)
        y=np.tanh(self.e[x].mean(1)@self.w.T+self.b)
        y=(y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)).astype(np.float32)
        results=[];offset=0
        for x in arrays:
            results.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor('EMBEDDINGS',y[offset:offset+len(x)].copy())]))
            offset+=len(x)
        return results
