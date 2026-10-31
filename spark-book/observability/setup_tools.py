"""Download pinned native Linux tools, verify hashes, and extract without installing."""
import argparse, hashlib, json, platform, subprocess, tarfile, urllib.request
from pathlib import Path

ASSETS = {
 'amd64': {
  'prometheus': ('https://github.com/prometheus/prometheus/releases/download/v3.15.1/prometheus-3.15.1.linux-amd64.tar.gz', '6a726885328ad9efebbf36e876ce2fb6225468a71fd1e7ce3ba666d5f4d97a84'),
  'grafana': ('https://github.com/grafana/grafana/releases/download/v13.3.0/grafana_13.3.0_40412481033_linux_amd64.deb', '458cb47824fda2c1e93c51e5d9c4893543f7fe24c58c2f5cb9c7a91d7360070f')},
 'arm64': {
  'prometheus': ('https://github.com/prometheus/prometheus/releases/download/v3.15.1/prometheus-3.15.1.linux-arm64.tar.gz', 'ceb6d21493c37ca0b4513b147642797c68cf39a0f50a444d50e1e80c74f21b26'),
  'grafana': ('https://github.com/grafana/grafana/releases/download/v13.3.0/grafana_13.3.0_40412481033_linux_arm64.deb', '6e687b1e4d8a3545035a030a4e7e0bc632ffdb483f1487f912fb0b3673853c30')}
}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 arch={'x86_64':'amd64','aarch64':'arm64','arm64':'arm64'}.get(platform.machine())
 if platform.system()!='Linux' or arch not in ASSETS:p.error('native Linux amd64 or arm64 required')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False,mode=0o700);receipt={'architecture':arch,'assets':{}}
 for name,(url,want) in ASSETS[arch].items():
  dest=out/url.rsplit('/',1)[1];digest=hashlib.sha256();count=0
  with urllib.request.urlopen(url,timeout=60) as response,dest.open('wb') as f:
   while data:=response.read(1024*1024):
    count+=len(data)
    if count>550*1024*1024:raise ValueError('asset byte budget exceeded')
    digest.update(data);f.write(data)
  if digest.hexdigest()!=want:raise ValueError('SHA256 mismatch for '+name)
  receipt['assets'][name]={'url':url,'bytes':count,'sha256':want}
  print(name+' SHA256 verified '+want,flush=True)
  if name=='prometheus':
   with tarfile.open(dest) as t:t.extractall(out,filter='data')
  else:subprocess.run(['dpkg-deb','--extract',str(dest),str(out/'grafana-package')],check=True)
 prom=out/('prometheus-3.15.1.linux-'+arch)
 home=out/'grafana-package/usr/share/grafana'
 receipt.update(prometheus=str(prom/'prometheus'),promtool=str(prom/'promtool'),grafana=str(home/'bin/grafana'),grafana_home=str(home))
 for tool,argv in [('prometheus',['--version']),('promtool',['--version']),('grafana',['server','--version'])]:
  result=subprocess.run([receipt[tool],*argv],capture_output=True,text=True)
  receipt[tool+'_version']={'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
  print(tool+' version_exit='+str(result.returncode),flush=True)
  if result.returncode:raise RuntimeError(tool+' did not execute natively; inspect runtime dependencies')
 (out/'tools.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('tools='+str(out/'tools.json'))
 print('Downloaded tools are not included in the companion archive. No package or driver installed.')
if __name__=='__main__':main()
