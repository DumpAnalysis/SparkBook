"""Optional native Docker Engine in the SAME WSL2 distro; not Docker Desktop qualification."""
import argparse,json,os,re,secrets
from pathlib import Path
from configure import configure,save
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--prom-image',required=True);p.add_argument('--grafana-image',required=True);a=p.parse_args()
 for repo,value in [('prom/prometheus',a.prom_image),('grafana/grafana',a.grafana_image)]:
  if not re.fullmatch(re.escape(repo)+r'@sha256:[a-f0-9]{64}',value):p.error(repo+' must be the qualified RepoDigest')
 a.out.mkdir(parents=True,exist_ok=False);a.out.chmod(0o700);out=configure(a.out)
 for name in ['prometheus.yml','provisioning/dashboards/lab.yaml']:
  path=out/name;path.write_text(path.read_text().replace(str(out),'/lab'))
 with open(out/'grafana-password.txt','x',opener=lambda path,flags:os.open(path,flags,0o600)) as f:f.write(secrets.token_urlsafe(24)+'\n')
 common={'network_mode':'host','user':f'{os.getuid()}:{os.getgid()}','read_only':True,'tmpfs':['/tmp'],'mem_limit':'1g','cpus':1,'logging':{'driver':'json-file','options':{'max-size':'10m','max-file':'2'}}}
 prom=dict(common,image=a.prom_image,volumes=[str(out)+':/lab:ro',str(out/'data-prometheus')+':/prometheus:rw'],command=['--config.file=/lab/prometheus.yml','--storage.tsdb.path=/prometheus','--storage.tsdb.retention.time=24h','--storage.tsdb.retention.size=500MB','--web.listen-address=127.0.0.1:9090'])
 graf=dict(common,image=a.grafana_image,volumes=[str(out)+':/lab:ro',str(out/'data-grafana')+':/var/lib/grafana:rw'],environment={'GF_PATHS_CONFIG':'/lab/grafana.ini','GF_PATHS_DATA':'/var/lib/grafana','GF_PATHS_LOGS':'/tmp','GF_PATHS_PLUGINS':'/var/lib/grafana/plugins','GF_PATHS_BUNDLED_PLUGINS':'/usr/share/grafana/data/plugins-bundled','GF_PATHS_PROVISIONING':'/lab/provisioning','GF_SECURITY_ADMIN_USER':'admin','GF_SECURITY_ADMIN_PASSWORD__FILE':'/lab/grafana-password.txt'})
 save(out/'compose.json',{'name':'wsl-gpu-observability','services':{'prometheus':prom,'grafana':graf}})
 print('Wrote '+str(out/'compose.json'));print('Run exporter and optional synthetic workload in this WSL distro; do not run the native stack simultaneously.')
if __name__=='__main__':main()
