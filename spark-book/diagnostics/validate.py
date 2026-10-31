import unittest,json
from pathlib import Path
import torch
from lab import numerical,compare,quality_run,STAGES,reduce_records,fails,sample_audit,trace_audit,weighted_mean,roof_models
ROOT=Path(__file__).resolve().parent
class Checks(unittest.TestCase):
 def test_01_float32(self):self.assertTrue(all(r[2] for r in compare(numerical(torch.float64),numerical(torch.float32))))
 def test_02_bfloat16(self):self.assertTrue(all(r[2] for r in compare(numerical(torch.float64),numerical(torch.bfloat16))))
 def test_03_first_boundary(self):self.assertEqual(next(n for n,e,p in compare(numerical(torch.float64),numerical(torch.float32,True)) if not p),'pooled')
 def test_04_argmax_misses_fault(self):self.assertTrue(torch.equal(numerical(torch.float32)['logits'].argmax(-1),numerical(torch.float32,True)['logits'].argmax(-1)))
 def test_05_nan(self):self.assertFalse(compare({'x':torch.ones(1)},{'x':torch.tensor([float('nan')])})[0][2])
 def test_06_shape(self):
  with self.assertRaises(ValueError):compare({'x':torch.ones(1)},{'x':torch.ones(2)})
 def test_07_ids_exact(self):self.assertFalse(compare({'ids':torch.tensor([1])},{'ids':torch.tensor([2])},100,100)[0][2])
 def test_08_near_zero(self):self.assertTrue(compare({'x':torch.tensor([0.])},{'x':torch.tensor([1e-7])},1e-6,0)[0][2])
 def test_09_quality_healthy(self):self.assertTrue(quality_run()['pass'])
 def test_10_quality_repairs(self):
  for fault in STAGES:
   self.assertFalse(quality_run(fault)['pass']);self.assertEqual([s for s in STAGES if quality_run(fault,s)['pass']],[fault])
 def test_11_failure_preserved(self):self.assertTrue(fails(reduce_records(['a','PAD','TRIGGER','b'],fails)))
 def test_12_one_minimal(self):
  r=reduce_records(['a','PAD','TRIGGER','b'],fails);self.assertTrue(all(not fails(r[:i]+r[i+1:]) for i in range(len(r))))
 def test_13_initial_pass(self):
  with self.assertRaises(ValueError):reduce_records(['x'],fails)
 def test_14_nonmonotonic(self):self.assertEqual([fails(['PAD','TRIGGER'],v) for v in range(1,6)],[False,False,True,False,True])
 def sample(self):return [{'id':i,'time':i,'age':0,'unit':'MiB'} for i in range(3)]
 def test_15_healthy_samples(self):self.assertEqual(sample_audit(self.sample()),[])
 def test_16_duplicate(self):r=self.sample();r.append(r[-1]);self.assertIn('duplicate_sample',sample_audit(r))
 def test_17_units(self):r=self.sample();r[0]['unit']='bytes';self.assertIn('unit_mismatch',sample_audit(r))
 def test_18_stale(self):r=self.sample();r[0]['age']=3;self.assertIn('stale_reading',sample_audit(r))
 def test_19_backwards(self):r=self.sample();r[2]['time']=-1;self.assertIn('backward_time',sample_audit(r))
 def test_20_weighting(self):self.assertEqual(weighted_mean([(1,100),(9,10)]),19)
 def test_21_trace(self):
  root={'span_id':'r','trace_id':'t','parent_id':None,'attributes':{'lab.expected_spans':2}};child={'span_id':'c','trace_id':'t','parent_id':'r','attributes':{}}
  self.assertEqual(trace_audit([root,child],[{'span_id':'c','trace_id':'t'}]),[]);self.assertIn('missing_or_extra_span',trace_audit([root],[]));self.assertIn('uncorrelated_log',trace_audit([root,child],[{'span_id':'c','trace_id':'wrong'}]))
 def test_22_roofline(self):
  m=roof_models();self.assertAlmostEqual(m[0][1]/m[0][2],1/12);self.assertGreater(m[2][1]/m[2][2],m[1][1]/m[1][2])
 def test_23_catalog(self):
  rows=json.loads((ROOT/'patterns.json').read_text());self.assertEqual(len(rows),12);self.assertEqual(len({r['name'] for r in rows}),12)
  self.assertTrue(all(all(r.get(k) for k in ['symptom','alternatives','evidence','lab','repair','regression','boundary']) for r in rows))
 def test_24_blind(self):
  c=json.loads((ROOT/'blind/cases.json').read_text());a=json.loads((ROOT/'blind/answers.json').read_text());self.assertEqual({r['id'] for r in c},{r['id'] for r in a});self.assertEqual(len(c),6);self.assertTrue(all(set(r)=={'id','evidence'} for r in c))
if __name__=='__main__':unittest.main(verbosity=2)
