import unittest
import numpy as np
from common import normalize,pool,metrics
from data import records,validate
class EmbeddingTests(unittest.TestCase):
    def test_padding_invariance(self):
        a=pool([[[1,2],[3,4]]],[[1,1]])
        b=pool([[[1,2],[3,4],[1000,-1000]]],[[1,1,0]])
        np.testing.assert_allclose(a,b)
    def test_bad_pooling_detected(self):
        good=pool([[[2,0],[0,4],[100,100]]],[[1,1,0]])
        self.assertGreater(np.linalg.norm(good-np.mean([[[2,0],[0,4],[100,100]]],axis=1)),10)
    def test_empty_pool(self):
        with self.assertRaises(ValueError):pool([[[1,2]]],[[0]])
    def test_zero_rejected(self):
        with self.assertRaises(ValueError):normalize([[0,0]])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):normalize([[np.nan,1]])
    def test_cosine_scale(self):
        np.testing.assert_allclose(normalize([[3,4]]),normalize([[30,40]]))
    def test_metric_exact(self):
        m=metrics([[1,0],[0,1]],[[1,0],[0,1]],[{0},{1}]);self.assertEqual(m['mrr'],1.)
    def test_metric_wrong(self):
        m=metrics([[1,0],[0,1]],[[1,0],[0,1]],[{1},{0}]);self.assertEqual(m['recall_at_1'],0.);self.assertEqual(m['mrr'],.5)
    def test_multiple_relevant(self):
        m=metrics([[1,0]],[[1,0],[.9,.1],[0,1]],[{0,1}]);self.assertEqual(m['recall_at_1'],.5)
    def test_family_disjoint(self):self.assertEqual(validate(),{'train':12,'dev':3,'test':3})
    def test_duplicate_family(self):
        with self.assertRaises(AssertionError):validate(records()+[records()[0]])
    def test_duplicate_text(self):
        rs=records();rs[1]['document']=rs[0]['document']
        with self.assertRaises(AssertionError):validate(rs)
if __name__=='__main__':unittest.main(verbosity=2)
