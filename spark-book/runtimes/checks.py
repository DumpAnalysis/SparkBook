"""Regression controls for the new evidence boundary, independent of GPU packages."""
import ast,json,tempfile,unittest,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import common
from lab import budget,summarize,split_check
class Checks(unittest.TestCase):
    def test_url_boundary(self):
        for u in ['http://127.0.0.1:80@evil:80','http://127.0.0.1:80/path','http://127.0.0.1:80?x=1','http://[::1]:80','file:///tmp/model']:
            with self.assertRaises(ValueError):common.endpoint(u,'/api/tags')
    def test_no_redirect_and_http_error_body(self):
        class H(BaseHTTPRequestHandler):
            calls=[]
            def log_message(self,*a):pass
            def do_GET(self):
                self.calls.append(self.path)
                self.send_response(302 if self.path=='/redirect' else 503)
                self.send_header('Location','/second');self.end_headers();self.wfile.write(b'original error')
        s=ThreadingHTTPServer(('127.0.0.1',0),H);t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
        try:
            base='http://127.0.0.1:'+str(s.server_port)
            with patch.dict('os.environ',{'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','NO_PROXY':''}):
                self.assertEqual(common.exchange(base,'/redirect'),(302,b'original error'))
                self.assertEqual(common.exchange(base,'/error'),(503,b'original error'))
            self.assertEqual(H.calls,['/redirect','/error'])
        finally:s.shutdown();s.server_close();t.join()
    def test_stream_failures(self):
        for b in [b'',b'{"done":false}',b'{"done":true}\n{"done":false}',b'{"done":true,"error":"oom"}']:
            with self.assertRaises(ValueError):common.ndjson(b)
        with self.assertRaises(ValueError):common.sse(b'data: [DONE]\n\n')
    def test_embeddings(self):
        for rows,count in [([[1],[2,3]],2),([[float('inf')]],1),([[0,0]],1),([[True]],1),([[1]],2)]:
            with self.assertRaises(ValueError):common.vectors(rows,count)
        self.assertAlmostEqual(common.cosine([1,0],[0,1]),0)
    def test_split_and_denominator(self):
        with self.assertRaises(ValueError):split_check([' CUDA OOM '],['cuda oom'])
        r=summarize([{'status':'error','complete':False,'seconds':1},{'status':'ok','complete':True,'seconds':3}])
        self.assertEqual((r['submitted'],r['completed'],r['failures'],r['median_seconds']),(2,1,1,3))
    def test_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a').write_text('a');x=common.tree_identity(p)[0];(p/'a').write_text('b');self.assertNotEqual(x,common.tree_identity(p)[0])
            with self.assertRaises(ValueError):common.fresh(p)
    def test_pinned_metadata(self):
        root=Path(__file__).parent
        for f in ['qwen-base','qwen-gguf','nemotron-base','nemotron-gguf']:
            m=json.loads((root/'metadata'/ (f+'.json')).read_text());self.assertRegex(m['sha'],r'^[a-f0-9]{40}$')
        m=json.loads((root/'metadata/nemotron-gguf.json').read_text());q=next(x for x in m['siblings'] if x['rfilename'].endswith('Q8_0.gguf'))
        self.assertFalse(budget(q['size'],30*1024**3,40*1024**3,6*1024**3)['gpu_weight_fit'])
    def test_target_syntax(self):
        for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text(),filename=f.name)
if __name__=='__main__':unittest.main(verbosity=2)
