"""Run one selected, locally qualified CUTLASS kernel and retain all evidence."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
from reference import save_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profiler',required=True)
    p.add_argument('--kernel',required=True,help='One exact operation name from this build')
    p.add_argument('--source-revision',required=True,help='40-character source commit')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{40}',a.source_revision):
        raise SystemExit('record the exact CUTLASS source commit')
    if not re.fullmatch(r'[A-Za-z0-9_]+',a.kernel):
        raise SystemExit('supply one exact kernel name without wildcards')
    exe=Path(a.profiler).resolve(strict=True)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    argv=[str(exe),'--operation=Gemm','--kernels='+a.kernel,
          '--m=512','--n=512','--k=512','--A=f16:row','--B=f16:row',
          '--C=f32:row','--accum=f32','--alpha=1','--beta=0',
          '--providers=cutlass','--verification-enabled=true',
          '--verification-providers=cublas','--warmup-iterations=5',
          '--profiling-iterations=20','--output='+str(out/'results')]
    with (out/'console.txt').open('w') as f:
        r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    text=(out/'console.txt').read_text()
    passed=bool(re.search(r'Disposition:\s+Passed',text))
    cublas=bool(re.search(r'cuBLAS:\s+Passed',text,re.IGNORECASE))
    verified=r.returncode==0 and passed and cublas
    with exe.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    save_new(out/'manifest.json',{'argv':argv,'source_revision':a.source_revision,
             'binary_sha256':digest,'returncode':r.returncode,
             'verified':verified,'note':'Inspect CSV kernel identity and numerical settings'})
    print('cutlass_cublas_verification='+('PASS' if verified else 'NOT_ESTABLISHED'))
    print('saved='+a.out)
    return 0 if verified else 2


if __name__=='__main__':
    raise SystemExit(main())
