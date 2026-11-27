"""Regression tests for evidence boundaries that affect the reader's acceptance decisions."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAB=Path(__file__).resolve().parents[1]/'spark-book'


class ReaderEvidence(unittest.TestCase):
    def test_generated_text_never_becomes_a_startup_observation(self):
        with tempfile.TemporaryDirectory(dir=LAB) as temporary:
            import shutil
            isolated=Path(temporary)
            for name in ('capture_evidence.py','spark_lab.py'):
                shutil.copyfile(LAB/name,isolated/name)
            runs=isolated/'runs';server=runs/'owned';server.mkdir(parents=True)
            (runs/'linux-server-state.json').write_text(json.dumps({'directory':'runs/owned'}))
            (runs/'receipt.json').write_text(json.dumps({'status':200,'body':{'choices':[{'message':{'content':'invented incident'}}]}}))
            (server/'stderr.txt').write_text('0.03.408.113 I srv  server_run_i: model loaded\n'
                '0.03.408.119 I srv  server_on_li: listening on http://127.0.0.1:8080\n'
                'prompt says srv server_run_i: invented incident\n')
            (server/'command.sh').write_text('llama-server -c 4096 -np 1 -ngl 99')
            subprocess.run([sys.executable,'capture_evidence.py','--request','runs/receipt.json',
                            '--out','runs/snapshot.json'],cwd=isolated,check=True,capture_output=True)
            snapshot=json.loads((runs/'snapshot.json').read_text())
            self.assertIn('HTTP status 200',snapshot['request'])
            self.assertIn('model loaded',snapshot['runtime'])
            self.assertNotIn('invented',json.dumps(snapshot))

    def test_guard_retains_partial_request_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);stamp='2026-11-27T12:00:00Z'
            for name,record in {
                'result':{'mode':'model','status':'BLOCKED','reason':'guard interrupted'},
                'admission':{'checks':{'guest_floor':False,'host_floor':True}},
                'requests':{'placement':[51,65],'requests':[{'label':'first','started_utc':stamp,'ended_utc':stamp}]}
            }.items():(run/(name+'.json')).write_text(json.dumps(record))
            (run/'host.jsonl').write_text(json.dumps({'utc':stamp,'available_bytes':12,'committed_bytes':3,'commit_limit_bytes':20})+'\n')
            done=subprocess.run([sys.executable,str(LAB/'memory_budget/guarded.py'),'report','--run',str(run)],text=True,capture_output=True)
            self.assertEqual(done.returncode,3,done.stderr)
            report=json.loads(done.stdout)
            self.assertEqual(report['failed_checks'],['guest_floor'])
            self.assertIsNone(report['pressure_request'])
            self.assertEqual(report['request_windows'][0]['host']['inside'][0]['commit_headroom_bytes'],17)


if __name__=='__main__':
    unittest.main()
