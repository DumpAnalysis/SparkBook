"""Use Triton's supported explicit library path; preserve injected driver bytes."""
from pathlib import Path
import ctypes,hashlib,json,os,sys

def configure():
 candidates=[p for p in Path('/usr/lib/wsl/drivers').glob('*/libcuda.so.1') if p.is_file()]
 if len(candidates)!=1:raise RuntimeError('Expected one actually injected WSL libcuda.so.1; found '+repr([str(p) for p in candidates]))
 driver=candidates[0];library=ctypes.CDLL(str(driver));version=ctypes.c_int();rc=library.cuDriverGetVersion(ctypes.byref(version));assert rc==0
 directory=Path('/tmp/spark-cuda-driver');directory.mkdir(exist_ok=True)
 for name in ['libcuda.so','libcuda.so.1']:
  link=directory/name
  if link.is_symlink():assert link.resolve()==driver.resolve()
  else:link.symlink_to(driver)
 os.environ['TRITON_LIBCUDA_PATH']=str(directory)
 with driver.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 print(json.dumps({'driver_library':str(driver),'resolved':str(driver.resolve()),'sha256':digest,'driver_API_version':version.value,'TRITON_LIBCUDA_PATH':str(directory),'change':'Only temporary symlink aliases to the existing injected driver; no replacement driver or global linker/cache changes'}),flush=True)
 return directory
if __name__=='__main__':
 configure()
 assert len(sys.argv)>1
 os.execvp(sys.argv[1],sys.argv[1:])
