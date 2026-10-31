"""Portable algorithm checks. These do not compile or execute CUDA code."""
import csv,math,pathlib,tempfile,unittest
import reference as r
import timing_summary as timing

class KernelReferenceTests(unittest.TestCase):
    def test_grid_coverage(self):
        for n in [1,31,32,33,255,256,257,1003]:
            for block in [128,256,512]:
                visits=[i for group in r.grid_indices(n,block,2) for i in group]
                self.assertEqual(sorted(visits),list(range(n)))
    def test_transpose_tail_producers(self):
        for rows,cols in [(1,65),(37,65),(32,32),(33,31),(65,37)]:
            mapping=r.transpose_mapping(rows,cols)
            self.assertEqual(mapping,{c*rows+row:row*cols+c for row in range(rows) for c in range(cols)})
    def test_two_pass_reduction(self):
        values=[i%17-8 for i in range(1003)]
        for block in [128,256,512]:
            partial=r.tree_sum(values,block,7)
            self.assertEqual(r.tree_sum(partial,block,1),[sum(values)])
        with self.assertRaises(ValueError):r.tree_sum(values,192,2)
    def test_histogram_partition(self):
        bins=[sum(i%16==b for i in range(1003)) for b in range(16)]
        self.assertEqual(bins,[63]*11+[62]*5);self.assertEqual(sum(bins),1003)
    def test_rectangular_matrix_product(self):
        self.assertEqual(r.matmul([[1,2,3],[4,5,6]],[[1,0],[0,1],[1,1]]),[[4,5],[10,11]])
    def test_rms_zero_and_scale(self):
        self.assertEqual(r.rms([0,0,0],[1,2,3]),[0,0,0])
        v=r.rms([1,-2,3],[1,1,1]);w=r.rms([1,-2,3],[2,2,2])
        for x,y in zip(v,w):self.assertAlmostEqual(2*x,y)
        self.assertTrue(all(math.isfinite(x) for x in v))
    def test_softmax_shift_and_normalization(self):
        a=r.softmax([0,1,2]);b=r.softmax([1000,1001,1002])
        self.assertAlmostEqual(sum(b),1)
        for x,y in zip(a,b):self.assertAlmostEqual(x,y)
        with self.assertRaises(ValueError):r.softmax([float('inf')])
    def test_timing_units_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'timing.csv';timing.fixture(path);rows=timing.read(path)
            self.assertAlmostEqual(rows[0]['mean_ms']/rows[1]['mean_ms'],5/3)
            with path.open() as f:data=list(csv.reader(f))
            data[1][4]='-1'
            with path.open('w',newline='') as f:csv.writer(f).writerows(data)
            with self.assertRaises(ValueError):timing.read(path)

if __name__=='__main__':
    print('Scope: CPU algorithm and timing-arithmetic checks; no CUDA execution.',flush=True)
    unittest.main(verbosity=2)
