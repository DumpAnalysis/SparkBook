"""Read a bounded uint32 buffer export or create an explicitly CPU fixture."""
import argparse,hashlib,json,pathlib,struct

def inspect(directory):
    root=pathlib.Path(directory).resolve()
    meta=json.loads((root/'metadata.json').read_text())
    if meta['dtype']!='uint32' or meta['byte_order']!='little':
        raise ValueError('Unsupported dtype or byte order')
    if type(meta['count']) is not int or not 0<meta['count']<=2**20:
        raise ValueError('Invalid or excessive element count')
    path=(root/meta['file']).resolve()
    if path.parent!=root:raise ValueError('Data file must be inside snapshot directory')
    if path.stat().st_size!=meta['count']*4:raise ValueError('Byte count mismatch')
    data=path.read_bytes();values=list(struct.unpack('<'+'I'*meta['count'],data))
    return meta,values,hashlib.sha256(data).hexdigest()

def fixture(directory):
    root=pathlib.Path(directory);root.mkdir(parents=True,exist_ok=False)
    values=[2*i+1 for i in range(256)]
    (root/'values.u32le').write_bytes(struct.pack('<256I',*values))
    (root/'metadata.json').write_text(json.dumps({'evidence':'CPU teaching fixture; no GPU',
        'dtype':'uint32','byte_order':'little','count':256,'file':'values.u32le',
        'stage':'synthetic post-transform values'},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--fixture',action='store_true')
    args=p.parse_args()
    if args.fixture:fixture(args.directory)
    meta,v,sha=inspect(args.directory)
    print('evidence='+meta['evidence']);print('bytes='+str(len(v)*4))
    print('first8='+','.join(map(str,v[:8])));print('last='+str(v[-1]))
    print('sum='+str(sum(v)));print('sha256='+sha)
    print('odd_sequence='+str(all(x==2*i+1 for i,x in enumerate(v))))
if __name__=='__main__':main()
