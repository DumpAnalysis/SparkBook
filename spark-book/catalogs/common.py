"""Small public-artifact receipt helpers; no credential configuration is read."""
import hashlib,json,re,time,urllib.request
from pathlib import Path,PurePosixPath

def sha(data):return hashlib.sha256(data).hexdigest()
def new_dir(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('unsafe relative filename')
    return p

def public_json(url,max_bytes=2*1024*1024):
    start=time.time_ns()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'spark-book-catalog-lab/19'}),timeout=60) as response:
        data=response.read(max_bytes+1)
        if len(data)>max_bytes:raise ValueError('metadata exceeds byte budget')
        receipt={'url':url,'status':response.status,'bytes':len(data),'sha256':sha(data),'start_ns':start,'end_ns':time.time_ns()}
    return json.loads(data),data,receipt

def download(url,path,expected_sha,maximum):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha):raise ValueError('expected SHA-256 required')
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.partial');count=0;digest=hashlib.sha256()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'spark-book-catalog-lab/19'}),timeout=60) as response,temp.open('xb') as stream:
            while chunk:=response.read(1024*1024):
                count+=len(chunk)
                if count>maximum:raise ValueError('download exceeds byte budget')
                digest.update(chunk);stream.write(chunk)
        if digest.hexdigest()!=expected_sha:raise ValueError('download SHA-256 mismatch')
        temp.rename(path)
    except BaseException:
        temp.unlink(missing_ok=True);raise
    return {'url':url,'bytes':count,'sha256':digest.hexdigest()}

def file_inventory(folder,names):
    root=Path(folder).resolve();files=[]
    for name in sorted(names):
        safe_name(name);p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise ValueError('missing or escaping artifact: '+name)
        b=p.read_bytes();files.append({'path':name,'bytes':len(b),'sha256':sha(b)})
    return files

def verify_inventory(folder,files):
    if not files:raise ValueError('empty inventory')
    expected={x['path'] for x in files}
    if len(expected)!=len(files):raise ValueError('duplicate inventory path')
    actual=file_inventory(folder,expected)
    if actual!=sorted(files,key=lambda x:x['path']):raise ValueError('artifact bytes differ from receipt')
    return len(actual)
