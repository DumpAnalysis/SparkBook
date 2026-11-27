"""Offline regression checks for reader evidence and interrupted pinned transfers."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

BOOK = Path(__file__).resolve().parents[1]/'spark-book'
sys.path.insert(0,str(BOOK/'runtimes'))
import acquire
from pinned_download import fetch_pinned

DATA = b'0123456789abcdef'*8
DIGEST = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def do_GET(self):
        headers=self.headers['Range']
        self.server.ranges.append(headers)
        start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',headers).groups())
        self.send_response(206)
        incorrect = self.server.mode=='wrong-range'
        self.send_header('Content-Range',f'bytes {start+int(incorrect)}-{end}/{len(DATA)}')
        self.send_header('Content-Length',str(end-start+1))
        self.end_headers()
        self.wfile.write(DATA[start:start+17] if self.server.mode=='short' else DATA[start:end+1])


@contextlib.contextmanager
def server():
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as http:
        http.mode='short';http.ranges=[]
        worker=threading.Thread(target=http.serve_forever);worker.start()
        try:yield http,f'http://127.0.0.1:{http.server_port}/weights'
        finally:http.shutdown();worker.join()


class ReaderTests(unittest.TestCase):
    def test_interrupted_transfer_resumes_at_exact_offset(self):
        with tempfile.TemporaryDirectory() as folder, server() as (http,url):
            path=Path(folder)/'model.gguf';partial=path.with_suffix('.gguf.partial')
            with self.assertRaises(OSError):fetch_pinned(url,path,len(DATA),sha256=DIGEST)
            self.assertEqual(partial.read_bytes(),DATA[:17]);self.assertFalse(path.exists())
            http.mode='wrong-range'
            with self.assertRaises(ValueError):fetch_pinned(url,path,len(DATA),sha256=DIGEST)
            self.assertEqual(partial.read_bytes(),DATA[:17])
            http.mode='complete'
            with mock.patch('pinned_download.CHUNK',32):
                self.assertEqual(fetch_pinned(url,path,len(DATA),sha256=DIGEST),DIGEST)
            self.assertIn('bytes=17-48',http.ranges);self.assertEqual(path.read_bytes(),DATA);self.assertFalse(partial.exists())
            count=len(http.ranges);self.assertEqual(fetch_pinned(url,path,len(DATA),sha256=DIGEST),DIGEST);self.assertEqual(len(http.ranges),count)

    def test_wrong_hash_is_never_published(self):
        with tempfile.TemporaryDirectory() as folder, server() as (http,url):
            path=Path(folder)/'model';http.mode='complete'
            with self.assertRaises(ValueError):fetch_pinned(url,path,len(DATA),sha256='0'*64)
            self.assertFalse(path.exists());self.assertEqual(path.with_suffix('.partial').read_bytes(),DATA)

    def test_planned_transfer_records_the_request_and_verifies_existing_files(self):
        entry = {'rfilename': 'weights', 'size': len(DATA), 'lfs': {'sha256': DIGEST}}
        plan = {'repository': 'local/fixture', 'revision': 'f' * 40,
                'bytes': len(DATA), 'files': [entry]}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(acquire, 'select', return_value=plan):
            out = Path(folder)
            acquire.acquire('Q4_K_M', out, False)
            self.assertFalse(json.loads((out/'acquisition-plan.json').read_text())['download_requested'])
            (out/'weights').write_bytes(DATA)
            acquire.acquire('Q4_K_M', out, True)
            self.assertTrue(json.loads((out/'acquisition-plan.json').read_text())['download_requested'])
            self.assertEqual(json.loads((out/'acquisition-receipt.json').read_text())['files'], {'weights': DIGEST})

    def test_offline_plan_pairing_and_existing_record(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'plan';acquire.acquire('Q4_K_M',out,False)
            plan=json.loads((out/'acquisition-plan.json').read_text());self.assertEqual(plan['bytes'],22421827488)
            with self.assertRaises(ValueError):acquire.acquire('Q8_0',out,False)
            self.assertFalse((out/'acquisition-receipt.json').exists())
            target=Path(folder)/'answer.json';target.write_text('{"status":200}')
            run=subprocess.run([sys.executable,str(BOOK/'spark_lab.py'),'query',str(BOOK/'prompt.txt'),'--base','http://127.0.0.1:1','--out',str(target)],text=True,capture_output=True)
            self.assertNotEqual(run.returncode,0);self.assertIn('Evidence already exists',run.stderr);self.assertEqual(target.read_text(),'{"status":200}')


if __name__=='__main__':unittest.main()
