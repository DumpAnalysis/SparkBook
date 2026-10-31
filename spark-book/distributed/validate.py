"""Regression checks over executed controls and independent arithmetic oracles."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch

E=Path(__file__).resolve().parent/"evidence"
def read(name):return json.loads((E/name).read_text())
def events(name,rank):return [json.loads(s) for s in (E/name/f"rank-{rank}.jsonl").read_text().splitlines()]

class Checks(unittest.TestCase):
    def test_01_two_worker_reference(self):
        r=read("01-train2/result.json");self.assertLess(r["max_reference_error"],1e-9)
    def test_02_four_worker_reference(self):
        r=read("02-train4/result.json");self.assertLess(r["max_reference_error"],1e-9)
    def test_03_replicas(self):
        for name in ["01-train2","02-train4"]:self.assertEqual(read(name+"/result.json")["max_replica_error"],0)
    def test_04_disjoint_global_batch(self):
        for name,world in [("01-train2",2),("02-train4",4)]:
            partitions=[next(e["examples"] for e in events(name,r) if e["event"]=="data_partition") for r in range(world)]
            self.assertEqual(sorted(sum(partitions,[])),list(range(8)))
    def test_05_same_global_objective(self):
        a=read("01-train2/result.json");b=read("02-train4/result.json")
        np.testing.assert_allclose(a["losses"],b["losses"],rtol=0,atol=1e-12)
    def test_06_unequal_token_weighting_counterexample(self):
        # Rank means 2 and 8 with token counts 1 and 3: naive rank mean is wrong.
        correct=(2*1+8*3)/4
        self.assertEqual(correct,6.5);self.assertEqual((2+8)/2,5);self.assertNotEqual((2+8)/2,correct)
        scaled=[2*2*1/4,8*2*3/4]
        self.assertEqual(sum(scaled)/2,correct)
    def test_07_resume_matches_uninterrupted(self):
        a=read("01-train2/result.json");b=read("09-checkpoint/resume/result.json")
        self.assertEqual(b["resumed_at"],2)
        np.testing.assert_allclose(a["losses"][2:],b["losses"],rtol=0,atol=1e-12)
        for phase,step in [("first",2),("resume",4)]:
            ck=torch.load(E/f"09-checkpoint/{phase}/checkpoint-0.pt",weights_only=True)
            self.assertEqual(ck["next_step"],step)
            self.assertTrue(all(int(s["step"])==step for s in ck["optimizer"]["state"].values()))
    def test_08_delay_propagates_to_peer_wait(self):
        e=read("04-slow/spans-0.json")
        wait=next(x for x in e if x["name"]=="gradient_average" and x["step"]==1)
        self.assertGreater((wait["end_ns"]-wait["start_ns"])/1e9,.6)
    def test_09_trace_preserves_spans(self):
        spans=read("04-slow/spans-0.json")+read("04-slow/spans-1.json")
        trace=read("05-plot/timeline.json")["traceEvents"]
        self.assertEqual(len(spans),len(trace));self.assertEqual(len(trace),41)
        self.assertTrue(all(s["end_ns"]>=s["start_ns"] for s in spans))
        self.assertEqual(sum(t["name"]=="injected_delay" for t in trace),1)
    def test_10_crash_is_not_a_timeout(self):
        r=read("06-crash/result.json");self.assertEqual(r["failure"]["kind"],"worker_exit")
        self.assertEqual(r["exitcodes"][1],17)
    def test_11_mismatch_operations_are_retained(self):
        r=read("07-mismatch/result.json")
        self.assertEqual({m["operation"] for m in r["failure"]["observed"]},{"mean","broadcast"})
    def test_12_timeout_names_missing_rank(self):
        r=read("08-timeout/result.json");self.assertEqual(r["failure"]["missing"],[1])
    def test_13_sharding_forward_and_gradients(self):
        r=read("10-shard/result.json");self.assertEqual(r["forward_and_gradient_errors"],[[0,0,0],[0,0,0]])
    def test_14_bad_shard_order_rejected(self):
        x=np.arange(12).reshape(3,4)/8;w=np.arange(24).reshape(4,6)/16
        correct=x@w;wrong=np.concatenate([x@w[:,3:],x@w[:,:3]],axis=1)
        self.assertFalse(np.allclose(correct,wrong,atol=1e-12,rtol=0))

if __name__=="__main__":unittest.main(verbosity=2)
