"""Rebuild only the recorded 0.8.1 wheel's metadata; installs nothing."""
import argparse,base64,csv,hashlib,io,json,struct,zipfile
from pathlib import Path
EXPECTED='4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f'
VERSION='0.8.1+sparkbook.1'

def rewrite(entries, expected_payload):
    """Pure metadata transformation, also tested with labelled synthetic inputs."""
    infos={n.split('/')[0] for n in entries if '.dist-info/' in n}
    if len(infos)!=1:raise ValueError('one distribution metadata directory required')
    old=infos.pop();new='nvidia_cusparselt_cu13-'+VERSION+'.dist-info'
    for name in entries:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe wheel member')
    wheel=entries[old+'/WHEEL'].decode();metadata=entries[old+'/METADATA'].decode()
    if 'Tag: py3-none-manylinux2014_sbsa' not in wheel or 'Version: 0.8.1\n' not in metadata:
        raise ValueError('input metadata does not match the recorded defect')
    wheel=wheel.replace('Root-Is-Purelib: true','Root-Is-Purelib: false').replace('manylinux2014_sbsa','manylinux2014_aarch64')
    metadata=metadata.replace('Version: 0.8.1\n','Version: '+VERSION+'\n',1)
    payload={n:hashlib.sha256(b).hexdigest() for n,b in entries.items() if not n.startswith(old+'/')}
    if payload!=expected_payload:raise ValueError('native/nonmetadata payload differs from the recorded input')
    elf=[];out={}
    for name,data in entries.items():
        if name==old+'/RECORD':continue
        if data.startswith(b'\x7fELF'):
            if len(data)<20 or data[4:6]!=b'\x02\x01' or struct.unpack('<H',data[18:20])[0]!=183:
                raise ValueError('only the recorded AArch64 ELF identity is supported')
            elf.append(name)
        target=name.replace(old+'/',new+'/',1) if name.startswith(old+'/') else name
        if name==old+'/WHEEL':data=wheel.encode()
        if name==old+'/METADATA':data=metadata.encode()
        out[target]=data
    if not elf:raise ValueError('no AArch64 ELF payload')
    record=io.StringIO();writer=csv.writer(record,lineterminator='\n')
    for n,b in sorted(out.items()):writer.writerow([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('='),len(b)])
    writer.writerow([new+'/RECORD','','']);out[new+'/RECORD']=record.getvalue().encode()
    return out,payload,elf

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('output directory must be new')
    with a.input.open('rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if source_hash!=EXPECTED:p.error('input is not the exact recorded publisher wheel; no output created')
    receipt=json.loads((Path(__file__).parent/'dependency-repair/repair.json').read_text())
    expected={r['path']:r['sha256'] for r in receipt['payload_before_install']}
    with zipfile.ZipFile(a.input) as z:
        if len(z.namelist())!=len(set(z.namelist())):p.error('duplicate wheel members')
        entries={i.filename:z.read(i) for i in z.infolist() if not i.is_dir()}
    out,payload,elf=rewrite(entries,expected)
    a.out.mkdir(parents=True)
    target=a.out/f'nvidia_cusparselt_cu13-{VERSION}-py3-none-manylinux2014_aarch64.whl'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:
        for n,b in sorted(out.items()):
            info=zipfile.ZipInfo(n,date_time=(1980,1,1,0,0,0));info.external_attr=0o644<<16;z.writestr(info,b)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in payload.items())
    with target.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(status='PASSED metadata reconstruction',input_sha256=source_hash,output=str(target.resolve()),output_sha256=h,local_version=VERSION,unchanged_payload=payload,elf_libraries=elf,installed=False,CUDA_executed=False,scope='Downstream metadata correction only; no ABI, sparse-kernel or full environment qualification')
    (a.out/'rebuild-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
