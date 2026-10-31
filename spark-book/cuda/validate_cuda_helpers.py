"""Portable artifact checks. No CUDA compilation, GPU, or profiler execution."""
import json,pathlib,sqlite3,tempfile,unittest
import dump_values,trace_sqlite

class EvidenceTests(unittest.TestCase):
    def test_expected_buffer(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'dump';dump_values.fixture(p)
            m,v,h=dump_values.inspect(p)
            self.assertEqual(v,list(range(1,512,2)))
            self.assertEqual(sum(v),65536)
            self.assertEqual(h,'0265543f1bc33584652f16078b7d9589242733c0febf55ef2ace28b89d484a8c')
            self.assertIn('no GPU',m['evidence'])
    def test_truncated_buffer(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'dump';dump_values.fixture(p)
            (p/'values.u32le').write_bytes(b'\x00')
            with self.assertRaisesRegex(ValueError,'Byte count'):dump_values.inspect(p)
    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'dump';dump_values.fixture(p)
            m=json.loads((p/'metadata.json').read_text());m['file']='../outside'
            (p/'metadata.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'inside'):dump_values.inspect(p)
    def test_trace_arithmetic(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'trace.sqlite';trace_sqlite.fixture(p)
            _,r=trace_sqlite.summarize(p)
            self.assertEqual(r,[('transform',2,500)])
    def test_missing_schema(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'trace.sqlite';sqlite3.connect(p).close()
            with self.assertRaisesRegex(ValueError,'tables absent'):trace_sqlite.summarize(p)
    def test_negative_duration(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'trace.sqlite';trace_sqlite.fixture(p)
            with sqlite3.connect(p) as con:con.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=0')
            with self.assertRaisesRegex(ValueError,'Negative'):trace_sqlite.summarize(p)
    def test_missing_name(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'trace.sqlite';trace_sqlite.fixture(p)
            with sqlite3.connect(p) as con:con.execute('DELETE FROM StringIds')
            with self.assertRaisesRegex(ValueError,'name IDs'):trace_sqlite.summarize(p)

if __name__=='__main__':
    print('Scope: CPU-created evidence fixtures; no GPU tools executed.',flush=True)
    unittest.main(verbosity=2)
