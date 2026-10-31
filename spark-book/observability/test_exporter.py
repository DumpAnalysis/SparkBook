import json,tempfile,unittest
from pathlib import Path
from exporter import parse,poll_fixture,render,FIELDS
from fixture import write
from configure import dashboard
from workload import Metrics
class ExporterTests(unittest.TestCase):
 def test_units(self):
  self.assertEqual(parse('memory.used','1'),(1048576.,'ok'));self.assertEqual(parse('clocks.current.sm','1'),(1000000.,'ok'))
 def test_zero(self):self.assertEqual(parse('utilization.gpu','0'),(0.,'ok'))
 def test_na(self):self.assertEqual(parse('utilization.gpu','N/A'),(None,'unavailable'))
 def test_nonfinite(self):
  for x in ['nan','inf','-inf']:self.assertEqual(parse('power.draw',x),(None,'parse_error'))
 def test_out_of_range(self):self.assertEqual(parse('utilization.gpu','101'),(None,'parse_error'))
 def test_negative(self):self.assertEqual(parse('memory.used','-1'),(None,'parse_error'))
 def test_multiple_gpus(self):self.assertEqual(parse('memory.used','1\n2'),(None,'shape_error'))
 def test_multiple_columns(self):self.assertEqual(parse('memory.used','1,2'),(None,'shape_error'))
 def test_malformed(self):self.assertEqual(parse('power.draw','watts'),(None,'parse_error'))
 def sample(self,case):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'input.json';write(path,case);fields,_=poll_fixture(path)
  return {'source':'fixture','gpu':'fixture-gpu','time':100.,'duration':.001,'fields':fields}
 def test_missing_omits_gauge(self):
  result=render(self.sample('missing'),now=101);self.assertNotIn('lab_gpu_busy_ratio{',result);self.assertIn('field="utilization.gpu"} 0',result)
 def test_zero_emits_gauge(self):self.assertIn('lab_gpu_busy_ratio{gpu="fixture-gpu",source="fixture"} 0.0',render(self.sample('zero'),now=101))
 def test_stale_omits_gauges(self):
  result=render(self.sample('baseline'),now=200);self.assertNotIn('lab_gpu_busy_ratio{',result);self.assertIn('status="stale"',result)
 def test_fixture_label_required(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'input.json';path.write_text('{}');fields,_=poll_fixture(path)
   self.assertTrue(all(x['value'] is None for x in fields.values()))
 def test_one_status_per_field(self):
  result=render(self.sample('baseline'),now=101);self.assertEqual(sum(x.startswith('lab_gpu_field_status{') for x in result.splitlines()),len(FIELDS))
 def test_dashboard_nulls_and_units(self):
  d=dashboard();self.assertEqual(len(d['panels']),13)
  self.assertTrue(all(not p['fieldConfig']['defaults']['custom']['spanNulls'] for p in d['panels'][1:]))
  self.assertTrue(all('or vector(0)' not in p['targets'][0]['expr'] for p in d['panels'][1:]))
 def test_cumulative_histogram(self):
  m=Metrics();m.observe(200,.02);m.observe(503,.2)
  self.assertEqual(m.buckets,[0,0,1,1,1,2,2,2]);self.assertEqual(m.n,2);self.assertAlmostEqual(m.total,.22)
if __name__=='__main__':unittest.main(verbosity=2)
