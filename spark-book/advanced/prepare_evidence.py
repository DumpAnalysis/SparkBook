"""Prepare fixed-key, bounded text evidence from four explicit local files."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from reference import save_new
from diagnostic_agent import FILES


def main():
    p=argparse.ArgumentParser()
    for key in FILES:p.add_argument('--'+key,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':{}}
    for key,name in FILES.items():
        path=Path(getattr(a,key)).resolve(strict=True)
        digest=hashlib.sha256()
        excerpt=b'';size=0
        with path.open('rb') as f:
            while True:
                chunk=f.read(65536)
                if not chunk:break
                digest.update(chunk);size+=len(chunk)
                if len(excerpt)<6000:excerpt+=chunk[:6000-len(excerpt)]
        # JSON is stored as evidence text, not reinterpreted as trusted policy.
        data=excerpt.decode('utf-8',errors='replace')
        metadata={'source':str(path),'sha256':digest.hexdigest(),
                  'bytes':size,'truncated':size>6000}
        text=json.dumps(metadata,separators=(',',':'))+'\n'+data
        # Bound encoded bytes even for replacement characters and long paths.
        text=text.encode('utf-8')[:8192].decode('utf-8',errors='ignore')
        (out/name).write_text(text,encoding='utf-8')
        manifest['sources'][key]=metadata
    save_new(out/'manifest.json',manifest)
    print('snapshots=4 read_only_keys=inventory,sanitizer,timeline,memory')
    print('saved='+a.out)


if __name__=='__main__':
    main()
