"""Public NGC metadata, verified ARM CLI download, and OCI manifest inspection."""
import argparse,base64,json,re,struct,urllib.error,urllib.parse,urllib.request,zipfile
from pathlib import Path
from common import new_dir,save,sha,public_json,download
LOCK=json.loads((Path(__file__).parent/'lock.json').read_text())
ACCEPT=', '.join(['application/vnd.oci.image.index.v1+json','application/vnd.docker.distribution.manifest.list.v2+json','application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json'])

def verify_digest(data,digest):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest) or 'sha256:'+sha(data)!=digest:raise ValueError('manifest digest mismatch')

def select_arm(index):
    choices=[x for x in index.get('manifests',[]) if x.get('platform',{}).get('os')=='linux' and x.get('platform',{}).get('architecture')=='arm64']
    if len(choices)!=1:raise ValueError(f'expected one linux/arm64 descriptor; found {len(choices)}')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',choices[0].get('digest','')):raise ValueError('invalid child digest')
    return choices[0]

class RegistryRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        target=super().redirect_request(req,fp,code,msg,headers,newurl)
        if target is not None and urllib.parse.urlparse(req.full_url).hostname!=urllib.parse.urlparse(newurl).hostname:
            target.remove_header('Authorization')
        return target

class Registry:
    def __init__(self):
        self.headers={'Accept':ACCEPT};self.receipts=[]
        self.opener=urllib.request.build_opener(RegistryRedirect())
    def get(self,path):
        url='https://nvcr.io'+path
        try:r=self.opener.open(urllib.request.Request(url,headers=self.headers),timeout=60)
        except urllib.error.HTTPError as error:
            if error.code!=401 or 'Authorization' in self.headers:raise
            challenge=error.headers.get('WWW-Authenticate','')
            if not challenge.lower().startswith('bearer '):raise ValueError('unsupported registry challenge')
            params=dict(re.findall(r'(realm|service|scope)="([^"]+)"',challenge));realm=urllib.parse.urlparse(params.get('realm',''))
            if realm.scheme!='https' or realm.hostname!='nvcr.io' or realm.username or realm.password:raise ValueError('unexpected registry authorization host')
            token_url=params['realm']+'?'+urllib.parse.urlencode({k:params[k] for k in ['service','scope'] if k in params})
            # Anonymous pull token remains in memory and is never retained or printed.
            with urllib.request.urlopen(token_url,timeout=60) as response:tokens=json.load(response)
            self.headers['Authorization']='Bearer '+tokens.get('token',tokens.get('access_token'))
            r=self.opener.open(urllib.request.Request(url,headers=self.headers),timeout=60)
        with r:
            b=r.read(2*1024*1024+1)
            if len(b)>2*1024*1024:raise ValueError('registry metadata exceeds byte budget')
            supplied=r.headers.get('Docker-Content-Digest')
            if supplied:verify_digest(b,supplied)
            self.receipts.append({'url':url,'status':r.status,'bytes':len(b),'sha256':sha(b)})
            return b,supplied or 'sha256:'+sha(b)

def info(out):
    out=new_dir(out);pin=LOCK['ngc_cli'];url=f"https://api.ngc.nvidia.com/v2/resources/{pin['resource']}/versions/{pin['version']}/files"
    data,raw,receipt=public_json(url);save(out/'http.json',receipt)
    projection={'recipe':{k:data['recipe'][k] for k in ['name','orgName','teamName','publisher','canGuestDownload']},'recipeVersion':{k:data['recipeVersion'][k] for k in ['versionId','status','totalSizeInBytes','totalFileCount']},'recipeFiles':data['recipeFiles'],'paginationInfo':data['paginationInfo'],'projection_note':'Human descriptions omitted; http.json hashes the complete response'}
    save(out/'resource.json',projection)
    if data['recipeVersion']['versionId']!=pin['version']:raise ValueError('wrong resource version')
    rows=data['recipeFiles'];arm=next(x for x in rows if x['path']==pin['arm64']['file'])
    decoded=base64.b64decode(arm['sha256_base64'],validate=True).hex()
    if decoded!=pin['arm64']['sha256'] or arm['sizeInBytes']!=pin['arm64']['bytes']:raise ValueError('published file differs from book pin')
    result={'resource':pin['resource'],'version':pin['version'],'publisher':data['recipe']['publisher'],'guest_download':data['recipe']['canGuestDownload'],'version_files':len(rows),'arm64_file':arm['path'],'arm64_bytes':arm['sizeInBytes'],'arm64_sha256':decoded}
    save(out/'summary.json',result)
    for k,v in result.items():print(f'{k}: {v}')

def cli(destination,record):
    out=new_dir(record);dest=new_dir(destination);pin=LOCK['ngc_cli'];a=pin['arm64']
    url=f"https://api.ngc.nvidia.com/v2/resources/{pin['resource']}/versions/{pin['version']}/files/{a['file']}"
    receipt=download(url,dest/a['file'],a['sha256'],a['bytes'])
    if receipt['bytes']!=a['bytes']:raise ValueError('archive size mismatch')
    with zipfile.ZipFile(dest/a['file']) as z:
        with z.open('ngc-cli/ngc') as f:header=f.read(64)
        if header[:4]!=b'\x7fELF' or header[4]!=2 or header[5]!=1:raise ValueError('expected little-endian ELF64 executable')
        machine=struct.unpack('<H',header[18:20])[0]
        if machine!=183:raise ValueError('archive is not ARM64')
        members=len(z.infolist())
    receipt.update(archive=str(dest/a['file']),elf_machine=machine,architecture='aarch64',zip_entries=members,executed=False)
    save(out/'receipt.json',receipt)
    print(f"archive: {a['file']}\nbytes: {receipt['bytes']}\nsha256: {receipt['sha256']}\nELF: 64-bit little-endian AArch64 (183)\nzip_entries: {members}\nSHA-256 PASS; binary was inspected, not executed")

def image(out):
    out=new_dir(out);tag=LOCK['ngc_pytorch'];repo,version=tag.removeprefix('nvcr.io/').rsplit(':',1);registry=Registry()
    index_raw,index_digest=registry.get(f'/v2/{repo}/manifests/{version}');index=json.loads(index_raw);selected=select_arm(index)
    child_raw,child_digest=registry.get(f"/v2/{repo}/manifests/{selected['digest']}");verify_digest(child_raw,selected['digest']);child=json.loads(child_raw)
    config_raw,_=registry.get(f"/v2/{repo}/blobs/{child['config']['digest']}");verify_digest(config_raw,child['config']['digest']);config=json.loads(config_raw)
    if (config.get('os'),config.get('architecture'))!=('linux','arm64'):raise ValueError('image config architecture mismatch')
    for name,b in [('index.json',index_raw),('manifest.json',child_raw),('config.json',config_raw)]: (out/name).write_bytes(b)
    result={'candidate':tag,'index_digest':index_digest,'arm64_digest':child_digest,'pinned_image':'nvcr.io/'+repo+'@'+child_digest,'os':'linux','architecture':'arm64','layers':len(child['layers']),'compressed_layer_bytes':sum(x['size'] for x in child['layers']),'layers_downloaded':False,'cuda_qualified':False}
    save(out/'image.json',result);save(out/'http.json',registry.receipts)
    for k in ['candidate','index_digest','arm64_digest','layers','compressed_layer_bytes']:print(f'{k}: {result[k]}')
    print('linux/arm64 metadata PASS; layers not downloaded; CUDA not qualified')

def faults(source,out):
    out=new_dir(out);raw=(Path(source)/'index.json').read_bytes();idx=json.loads(raw);passed=[]
    cases={
      'missing_arm64':dict(idx,manifests=[m for m in idx['manifests'] if m.get('platform',{}).get('architecture')!='arm64']),
      'duplicate_arm64':dict(idx,manifests=[select_arm(idx),select_arm(idx)])}
    for name,case in cases.items():
        try:select_arm(case)
        except ValueError as e:print(name+': REJECTED '+str(e));passed.append(name)
        else:raise AssertionError(name)
    try:verify_digest(raw+b' ',json.loads((Path(source)/'image.json').read_text())['index_digest'])
    except ValueError as e:print('changed_manifest: REJECTED '+str(e));passed.append('changed_manifest')
    else:raise AssertionError('modified manifest accepted')
    save(out/'result.json',{'passed':passed,'scope':'mutations of retained registry metadata; no Docker execution'})
    print('3 fault controls passed; no Docker or GPU execution')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['info','image']:
        q=sub.add_parser(name);q.add_argument('--out',required=True)
    q=sub.add_parser('cli');q.add_argument('--dest',required=True);q.add_argument('--record',required=True)
    q=sub.add_parser('faults');q.add_argument('--source',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='info':info(a.out)
    elif a.command=='cli':cli(a.dest,a.record)
    elif a.command=='image':image(a.out)
    else:faults(a.source,a.out)
if __name__=='__main__':main()
