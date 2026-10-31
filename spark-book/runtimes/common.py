"""Bounded loopback HTTP and evidence contracts; Python standard library only."""
import hashlib, json, math, urllib.request, urllib.error, urllib.parse
from pathlib import Path
MAX_BYTES = 4 * 1024 * 1024
class ContractError(ValueError): pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None

def endpoint(base, path):
    u = urllib.parse.urlsplit(base)
    if u.scheme != 'http' or u.hostname != '127.0.0.1' or u.username or u.password or u.query or u.fragment or u.path not in ('', '/'):
        raise ContractError('only an explicit http://127.0.0.1:port base is allowed')
    if not u.port or not 1 <= u.port <= 65535 or not path.startswith('/') or path.startswith('//'):
        raise ContractError('invalid endpoint')
    return base.rstrip('/') + path

def exchange(base, path, payload=None):
    request = urllib.request.Request(endpoint(base,path), data=None if payload is None else json.dumps(payload,allow_nan=False).encode(), headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(request,timeout=120)
    except urllib.error.HTTPError as exc: response=exc
    with response:
        body=response.read(MAX_BYTES+1)
        if len(body)>MAX_BYTES: raise ContractError('response exceeds byte limit')
        return response.code, body

def ndjson(body):
    rows=[json.loads(line) for line in body.splitlines() if line.strip()]
    if not rows or rows[-1].get('done') is not True or any(x.get('done') is True for x in rows[:-1]):
        raise ContractError('missing or misplaced terminal event')
    if any('error' in x for x in rows): raise ContractError('server error event')
    return rows

def sse(body):
    # Accept the data-only OpenAI chat stream used by this lab, not arbitrary SSE.
    events=[]; terminal=False
    for frame in body.decode().replace('\r\n','\n').split('\n\n'):
        fields=[x[5:].lstrip() for x in frame.splitlines() if x.startswith('data:')]
        if not fields: continue
        data='\n'.join(fields)
        if terminal: raise ContractError('event after DONE')
        if data=='[DONE]': terminal=True
        else: events.append(json.loads(data))
    if not terminal or not events: raise ContractError('missing SSE completion')
    if any('error' in e for e in events): raise ContractError('SSE error event')
    if not any(c.get('finish_reason') is not None for e in events for c in e.get('choices',[])):
        raise ContractError('missing finish reason')
    return events

def vectors(rows,count):
    if len(rows)!=count or not rows or not rows[0]: raise ContractError('vector count or empty vector')
    dim=len(rows[0])
    if any(len(r)!=dim or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in r) or sum(x*x for x in r)==0 for r in rows):
        raise ContractError('invalid vector shape, value or norm')
    return rows

def cosine(a,b):
    vectors([a,b],2)
    return sum(x*y for x,y in zip(a,b))/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def tree_identity(root):
    root=Path(root)
    records={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    if not records:raise ContractError('empty model directory')
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),records

def fresh(path):
    p=Path(path)
    if p.exists():raise ContractError('output already exists: '+str(p))
    p.mkdir(parents=True)
    return p

def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
