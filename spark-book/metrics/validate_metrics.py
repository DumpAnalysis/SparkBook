"""Portable numerical, missing-data, timing and process-lifecycle checks."""
import contextlib,io,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from monitor import FIELDS,value,parse,call,guest_memory,validate_plan,capture,run_phase
from plot import break_gaps,load,series,summarize
from fixture import create
from compare import compare

class MetricsChecks(unittest.TestCase):
    def test_zero_is_observed(self):
        self.assertEqual(value('utilization.gpu','0'),{'value':0.,'status':'ok'})
    def test_na_is_not_zero(self):
        self.assertEqual(value('memory.used','N/A'),{'value':None,'status':'unavailable'})
    def test_bracketed_unavailable(self):
        self.assertEqual(value('power.draw','[Not Supported]')['status'],'unavailable')
    def test_nonfinite_rejected(self):
        for x in ['nan','inf','-inf']:self.assertEqual(value('memory.used',x)['status'],'parse_error')
    def test_percent_range(self):
        for x in ['-1','101']:self.assertEqual(value('utilization.gpu',x)['status'],'parse_error')
    def test_csv_shape(self):
        self.assertEqual(parse(['memory.used'],'1\n2')['memory.used']['status'],'shape_error')
    def test_csv_partial_unavailable(self):
        x=parse(['memory.used','power.draw'],' 1024, N/A\n');self.assertEqual(x['memory.used']['value'],1024);self.assertIsNone(x['power.draw']['value'])
    def test_units_not_silently_stripped(self):
        self.assertEqual(value('memory.used','10 MiB')['status'],'parse_error')
    def test_query_launch_failure(self):
        self.assertEqual(call(['/book-test-missing-executable'])['error'],'launch_error')
    def test_query_timeout(self):
        x=call([sys.executable,'-c','import time; time.sleep(2)'],timeout=.05);self.assertEqual(x['error'],'timeout')
    def test_gap_is_broken(self):
        x,y=break_gaps([0,1,5],[2,3,4],2.5);self.assertEqual(x,[0,1,3,5]);self.assertTrue(math.isnan(y[2]))
    def test_plan_rejects_shell_string(self):
        with self.assertRaises(ValueError):validate_plan([{'label':'bad','argv':'echo test'}])
    def test_plan_timeout_bound(self):
        with self.assertRaises(ValueError):validate_plan([{'label':'bad','argv':['true'],'timeout':float('inf')}])
    def test_guest_units(self):
        with tempfile.TemporaryDirectory() as x:
            p=Path(x)/'mem';p.write_text('MemTotal: 2048 kB\nMemAvailable: 1024 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n');self.assertEqual(guest_memory(p)['MemTotal'],2.)
    def test_fixture_missingness_and_phase_spans(self):
        with tempfile.TemporaryDirectory() as x,contextlib.redirect_stdout(io.StringIO()):
            p=Path(x)/'fixture';create(p);m,r,e=load(p);s=summarize(m,r,e)
            self.assertEqual(len(r),43);self.assertEqual(s['fields']['power.draw']['valid'],40)
            self.assertEqual(s['fields']['clocks.current.memory']['sample_median'],None)
            self.assertEqual(s['phases'][0]['contained_query_spans'],5)
    def test_duplicate_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as x,contextlib.redirect_stdout(io.StringIO()):
            p=Path(x)/'fixture';create(p);q=p/'samples.jsonl';s=q.read_text();q.write_text(s+s.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):load(p)
    def test_timeout_stops_owned_workload(self):
        with tempfile.TemporaryDirectory() as x:
            e=run_phase({'label':'timeout','argv':[sys.executable,'-c','import time; time.sleep(10)'],'timeout':.05},Path(x),0,__import__('time').monotonic_ns());self.assertEqual(e['returncode'],124)
    def test_capture_keeps_failed_phase(self):
        with tempfile.TemporaryDirectory() as x,contextlib.redirect_stdout(io.StringIO()):
            p=Path(x)/'run';plan=[{'label':'fails','argv':[sys.executable,'-c','raise SystemExit(7)']},{'label':'skipped','argv':['true']}]
            code=capture(p,plan,.1,no_gpu=True);self.assertEqual(code,1)
            events=json.loads((p/'events.json').read_text());self.assertEqual(len(events),1);self.assertEqual(events[0]['returncode'],7)
    def test_no_gpu_control_cannot_invent_gpu_values(self):
        with tempfile.TemporaryDirectory() as x,contextlib.redirect_stdout(io.StringIO()):
            p=Path(x)/'run';capture(p,[{'label':'sleep','argv':[sys.executable,'-c','import time;time.sleep(.25)']}],.1,no_gpu=True)
            m,r,e=load(p);self.assertTrue(all(v['value'] is None and v['status']=='disabled' for z in r for v in z['gpu'].values()))
    def test_run_placeholder_retains_separate_results(self):
        with tempfile.TemporaryDirectory() as x:
            p=Path(x)
            e=run_phase({'label':'write','argv':[sys.executable,'-c',"import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ok')",'{run}/result.txt']},p,0,__import__('time').monotonic_ns())
            self.assertEqual(e['returncode'],0);self.assertEqual((p/'result.txt').read_text(),'ok')
            self.assertEqual(e['argv'][-1],str(p.resolve()/'result.txt'))
    def test_overhead_control_rejects_different_plans(self):
        with self.assertRaises(ValueError):compare([({'plan':[1]},[])],[({'plan':[2]},[])])

if __name__=='__main__':unittest.main(verbosity=2)
