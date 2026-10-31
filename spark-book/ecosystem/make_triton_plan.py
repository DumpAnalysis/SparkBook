"""Package a value-checked TensorRT target result; does not start Triton."""
import argparse,json,shutil
from pathlib import Path
from lab import sha,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.engine_dir/'tensorrt.json').read_text());plan=a.engine_dir/'linear.plan'
    assert report['plan_sha256']==sha(plan)
    assert [x['batch'] for x in report['samples']]==[1,4,8]
    assert all(x['max_error']<1e-3 for x in report['samples']) and report['out_of_profile_rejected']
    a.out.mkdir(parents=True,exist_ok=False);model=a.out/'lab_linear';(model/'1').mkdir(parents=True)
    shutil.copy2(plan,model/'1/model.plan')
    (model/'config.pbtxt').write_text('''name: "lab_linear"
platform: "tensorrt_plan"
max_batch_size: 8
input [{ name: "X" data_type: TYPE_FP32 dims: [8] }]
output [{ name: "Y" data_type: TYPE_FP32 dims: [4] }]
instance_group [{ count: 1 kind: KIND_GPU }]
version_policy { specific { versions: [1] } }
''')
    dump(a.out/'engine-provenance.json',report)
    print('Packaged lab_linear version 1; Triton execution still requires qualification')

if __name__=='__main__':main()
