"""Portable tests only. Does not test WSL2, a GPU, or a real model."""
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import agent
import spark_lab as lab
import finetune
import training_data


class StreamingFixture(lab.Fixture):
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        json.loads(self.rfile.read(size))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        objects = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'READY'}}]},
            {'choices': [], 'usage': {'completion_tokens': 1}}
        ]
        for obj in objects:
            self.wfile.write(('data: ' + json.dumps(obj) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')


class Tests(unittest.TestCase):
    def test_training_split_identity(self):
        ids=[key for rows in training_data.CASES.values() for key,_,_ in rows]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),20)

    def test_assistant_only_mask(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return [1,2,3] if len(messages)==2 else [1,2,3,4,5]
        row={'messages':[{}, {}, {}]}
        encoded=finetune.encode(row,Tokenizer())
        self.assertEqual(encoded['labels'],[-100,-100,-100,4,5])

    def test_training_overlength_rejected(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return [1,2,3] if len(messages)==2 else [1,2,3,4,5]
        with self.assertRaises(ValueError):
            finetune.encode({'messages':[{}, {}, {}]},Tokenizer(),4)

    def test_budget(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lab.budget(None)
        self.assertIn('context= 4096 KV=  224.0 MiB', out.getvalue())
        self.assertIn('30 decimal GB = 27.94 GiB', out.getvalue())

    def test_loopback_boundary(self):
        with self.assertRaises(ValueError):
            lab.http('https://example.com', '/health')

    def test_schema_good_read(self):
        action, result = agent.parse('{"action":"read","key":"limits"}', set())
        self.assertEqual(result['key'], 'limits')

    def test_schema_forbidden_tool(self):
        with self.assertRaises(ValueError):
            agent.parse('{"action":"shell","command":"anything"}', set())

    def test_schema_extra_argument(self):
        with self.assertRaises(ValueError):
            agent.parse('{"action":"read","key":"limits","path":"x"}', set())

    def test_schema_unread_citation(self):
        with self.assertRaises(ValueError):
            agent.parse('{"action":"final","diagnosis":"x","evidence":["limits"]}', set())

    def test_schema_good_final(self):
        obj, result = agent.parse(
            '{"action":"final","diagnosis":"x","evidence":["limits"]}', {'limits'})
        self.assertIsNone(result)

    def test_grounding(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp) / 'prompt.txt'
            with contextlib.redirect_stdout(io.StringIO()):
                lab.ground(SimpleNamespace(corpus='corpus.json',
                    question='Why did R17 time out waiting for service B?', out=str(out)))
            text = out.read_text()
            self.assertIn('[E1]', text)
            self.assertIn('[E2]', text)
            self.assertNotIn('[E3]', text)

    def test_http_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp) / 'http.json'
            with contextlib.redirect_stdout(io.StringIO()):
                lab.fixture(SimpleNamespace(out=str(out)))
            result = json.loads(out.read_text())
            self.assertEqual(result['wrong_route']['status'], 404)
            self.assertEqual(result['wrong_model']['status'], 404)

    def test_replay_and_denial(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, code in [('agent-actions.json', 0), ('agent-denied.json', 2)]:
                dest = pathlib.Path(temp) / (name + '.out')
                result = subprocess.run([sys.executable, 'agent.py',
                    '--replay', name, '--out', str(dest)], capture_output=True)
                self.assertEqual(result.returncode, code)
                self.assertTrue(json.loads(dest.read_text())['receipts'])

    def test_stream_parser(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), StreamingFixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                prompt = pathlib.Path(temp) / 'prompt.txt'
                prompt.write_text('Reply READY')
                dest = pathlib.Path(temp) / 'stream.json'
                result = subprocess.run([sys.executable, 'stream.py', str(prompt),
                    '--base', 'http://127.0.0.1:' + str(server.server_port),
                    '--out', str(dest)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(dest.read_text())
                self.assertEqual(record['content'], 'READY')
                self.assertEqual(len(record['events']), 4)
                self.assertIsNotNone(record['first_content_s'])
                self.assertTrue(record['done'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_elf_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / 'fixture.elf'
            data = bytearray(20)
            data[:6] = b'\x7fELF\x02\x01'
            data[18:20] = (183).to_bytes(2, 'little')
            path.write_bytes(data)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                lab.inspect(SimpleNamespace(path=str(path)))
            self.assertIn('kind=ELF aarch64', out.getvalue())

    def test_bad_response_is_not_answer(self):
        self.assertEqual(lab.answer({'status': 200, 'body': 'not json'}), (None, None))

    def test_summary_fixture(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lab.summarize(SimpleNamespace(path='example-benchmark.json'))
        self.assertIn('median_seconds=1.200', out.getvalue())
        self.assertIn('aggregate_output_tokens_per_second=83.333', out.getvalue())


if __name__ == '__main__':
    print('Validation scope: portable helpers; no model or GPU execution.', flush=True)
    unittest.main(verbosity=2)
