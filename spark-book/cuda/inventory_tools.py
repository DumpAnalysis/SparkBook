"""Inventory tools without treating presence as platform qualification."""
import datetime,json,pathlib,platform,shutil,subprocess
TOOLS=['nvcc','cuobjdump','nvdisasm','cuda-gdb','compute-sanitizer','nsys','ncu']

def main():
    root=pathlib.Path('runs/cuda');root.mkdir(parents=True,exist_ok=True)
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'machine':platform.machine(),'kernel':platform.release(),'tools':{}}
    for name in TOOLS:
        path=shutil.which(name)
        entry={'path':path,'qualified':False}
        if path:
            try:
                r=subprocess.run([path,'--version'],capture_output=True,text=True,timeout=15)
                entry.update(exit=r.returncode,output=r.stdout,error=r.stderr)
            except (OSError,subprocess.TimeoutExpired) as e:entry['error']=str(e)
        result['tools'][name]=entry
        print(name+'='+('present' if path else 'missing'))
    (root/'tools.json').write_text(json.dumps(result,indent=2))
    print('saved=runs/cuda/tools.json')
    print('Presence is not a successful GPU capability test.')
if __name__=='__main__':main()
