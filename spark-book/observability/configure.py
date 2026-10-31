"""Generate local Prometheus/Grafana files and a provisioned dashboard."""
import argparse,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def dashboard():
 source='source=~"$source",gpu=~"$gpu"';sel='{'+source+'}'
 spec=[
 ('GPU busy ratio','lab_gpu_busy_ratio'+sel,'percentunit'),
 ('GPU memory bytes','lab_gpu_memory_used_bytes'+sel,'bytes'),
 ('GPU memory pressure','lab:gpu_memory_used:ratio'+sel,'percentunit'),
 ('GPU temperature','lab_gpu_temperature_celsius'+sel,'celsius'),
 ('GPU power','lab_gpu_power_watts'+sel,'watt'),
 ('GPU SM clock','lab_gpu_sm_clock_hertz'+sel,'hertz'),
 ('Field availability: 0 means unavailable','lab_gpu_field_available'+sel,'short'),
 ('Scrape health: 1 means endpoint reachable','up{job="gpu"}','short'),
 ('Collector age','time() - lab_exporter_last_poll_timestamp_seconds','s'),
 ('SYNTHETIC CPU requests per second','sum(rate(lab_demo_requests_total[1m]))','reqps'),
 ('SYNTHETIC CPU p95 handler latency','histogram_quantile(0.95,sum by(le)(rate(lab_demo_request_duration_seconds_bucket[1m])))','s'),
 ('SYNTHETIC CPU error fraction','sum(rate(lab_demo_requests_total{status="503"}[1m])) / sum(rate(lab_demo_requests_total[1m]))','percentunit')]
 panels=[{'id':1,'type':'text','title':'Evidence and interpretation','gridPos':{'x':0,'y':0,'w':24,'h':3},'options':{'mode':'markdown','content':'**Select source `fixture` for SIMULATED values or `wsl` for live mapped-driver observations.** No-data is not zero. GPU byte totals are device-reported; 64 GB host memory is not a 64 GB GPU allocation budget. Synthetic CPU request panels are pipeline exercises, not model measurements. Rate and p95 panels need several scrapes and requests.'}}]
 for i,(title,expr,unit) in enumerate(spec):
  panels.append({'id':i+2,'type':'timeseries','title':title,'datasource':{'type':'prometheus','uid':'lab-prometheus'},'gridPos':{'x':(i%2)*12,'y':3+(i//2)*8,'w':12,'h':8},'targets':[{'refId':'A','expr':expr,'legendFormat':'{{field}} {{gpu}} {{instance}}'}],'fieldConfig':{'defaults':{'unit':unit,'custom':{'spanNulls':False}},'overrides':[]},'options':{'legend':{'displayMode':'list','placement':'bottom'}}})
 return {'uid':'wsl-gpu-lab','title':'WSL2 GPU laboratory - measured and simulated sources','schemaVersion':39,'version':1,'editable':False,'tags':['wsl2','laboratory'],'timezone':'browser','refresh':'5s','time':{'from':'now-15m','to':'now'},'templating':{'list':[{'name':'source','type':'custom','query':'fixture,wsl','current':{'text':'fixture','value':'fixture'},'options':[{'text':'fixture','value':'fixture','selected':True},{'text':'wsl','value':'wsl','selected':False}]},{'name':'gpu','type':'query','datasource':{'type':'prometheus','uid':'lab-prometheus'},'query':'label_values(lab_gpu_field_available{source=~"$source"},gpu)','refresh':1,'includeAll':True,'allValue':'.*','current':{'text':'All','value':'$__all'}}]},'panels':panels}
def configure(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 for name in ('data-prometheus','data-grafana','logs','provisioning/datasources','provisioning/dashboards','provisioning/plugins','provisioning/alerting','dashboards'):(out/name).mkdir(parents=True,exist_ok=True)
 shutil.copyfile(HERE/'rules.yml',out/'rules.yml')
 jobs=[{'job_name':'gpu','static_configs':[{'targets':['127.0.0.1:9400']}]},{'job_name':'synthetic-cpu','static_configs':[{'targets':['127.0.0.1:8089']}]},{'job_name':'prometheus','static_configs':[{'targets':['127.0.0.1:9090']}]},{'job_name':'inference','file_sd_configs':[{'files':[str(out/'inference-targets.json')],'refresh_interval':'5s'}]}]
 save(out/'prometheus.yml',{'global':{'scrape_interval':'5s','scrape_timeout':'4s','evaluation_interval':'5s'},'rule_files':[str(out/'rules.yml')],'scrape_configs':jobs})
 save(out/'inference-targets.json',[])
 save(out/'provisioning/datasources/prometheus.yaml',{'apiVersion':1,'datasources':[{'name':'Laboratory Prometheus','uid':'lab-prometheus','type':'prometheus','access':'proxy','url':'http://127.0.0.1:9090','isDefault':True,'editable':False,'jsonData':{'timeInterval':'5s'}}]})
 save(out/'provisioning/dashboards/lab.yaml',{'apiVersion':1,'providers':[{'name':'Laboratory','folder':'WSL2 Laboratory','type':'file','disableDeletion':True,'updateIntervalSeconds':10,'options':{'path':str(out/'dashboards'),'foldersFromFilesStructure':False}}]})
 save(out/'dashboards/gpu.json',dashboard())
 (out/'grafana.ini').write_text('[server]\nhttp_addr = 127.0.0.1\nhttp_port = 3000\n[paths]\ndata = '+str(out/'data-grafana')+'\nlogs = '+str(out/'logs')+'\nprovisioning = '+str(out/'provisioning')+'\n[users]\nallow_sign_up = false\n[auth.anonymous]\nenabled = false\n[analytics]\nreporting_enabled = false\ncheck_for_updates = false\ncheck_for_plugin_updates = false\n[plugins]\npreinstall_disabled = true\n[log]\nmode = console\nlevel = warn\n')
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();out=configure(a.out);print('Configured loopback stack at '+str(out))
