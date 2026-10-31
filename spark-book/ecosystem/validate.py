"""Evidence assertions, independent of target GPU availability."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from lab import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();r=a.evidence
    load=lambda f:json.loads((r/f).read_text());checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    fixture=np.load(r/'fixture/fixture.npz');identity=load('fixture/identity.json')
    analytics=load('analytics/analytics.json');ort=load('ort/ort.json');q=load('quantize/quantization.json')
    data=load('data/audit.json');lora=load('lora/lora.json');inv=load('inventory/inventory.json')
    check('fixture identity',identity['fixture_sha256']==sha(r/'fixture/fixture.npz'))
    check('requests retained',analytics['rows']==28)
    check('rejections excluded from mean',analytics['accepted']==24)
    check('both route means',all(x['count']==12 and x['mean_ms']==15.75 for x in analytics['groups']))
    proj=np.load(r/'analytics/pca.npz')['projection']
    check('PCA projector symmetry',np.allclose(proj,proj.T,atol=1e-6))
    check('PCA projector idempotence',np.allclose(proj@proj,proj,atol=1e-5))
    check('PCA rank two',abs(np.trace(proj)-2)<1e-5)
    check('ONNX valid',onnx.checker.check_model(str(r/'onnx/linear.onnx')) is None)
    check('ORT observed CPU',ort['observed']==['CPUExecutionProvider'])
    check('ORT all value comparisons',ort['max_error']<1e-5)
    check('ORT invalid width and dtype',ort['rejections']==['width','dtype'])
    check('QDQ valid',onnx.checker.check_model(str(r/'quantize/linear-qdq.onnx')) is None)
    check('QDQ CPU numerical oracle',all(v['ort_vs_numpy']<1e-5 for v in q['metrics'].values()))
    check('held-out unclipped control',q['metrics']['test']['clipped']==0)
    check('outlier detected',q['metrics']['shifted']['clipped']==1 and q['metrics']['shifted']['max_error']>1)
    check('per-channel preserves small weights',q['per_channel_small_mae']<q['per_tensor_small_mae'])
    check('data split and leakage injection',data['overlap']==0 and data['injected_overlap_detected']==1)
    check('shifted mask includes answer and EOS',data['shifted_supervised_tokens']==data['answer_including_eos'])
    check('frozen base and adapter restore',lora['frozen_base'] and lora['restore_max_error']<1e-5 and lora['final_loss']<lora['initial_loss'])
    check('every control completed',len(load('commands.json'))==8 and all(x['exit']==0 and (r/(x['name']+'.txt')).read_text().endswith('[exit 0]\n') for x in load('commands.json')))
    result=dict(checks=len(checks),passed=True,names=checks,
      target_gpu_frameworks_executed=False,scope='CPU numerical and evidence controls; no RAPIDS, TensorRT or NeMo execution')
    (r/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(checks)} CPU/evidence checks passed; target GPU qualification remains pending')

if __name__=='__main__':main()
