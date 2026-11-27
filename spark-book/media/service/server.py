"""A single CUDA MMS owner serving short text on guest loopback."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.io import wavfile
import torch
from transformers import AutoTokenizer, VitsModel

OUTPUT = Path('/output')
owner = threading.Lock()


def synthesize(text):
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0].float().cpu().numpy()
    if not waveform.size or not np.isfinite(waveform).all() or not np.abs(waveform).max():
        raise ValueError('Generated audio failed the signal check')
    return waveform


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            return self.reply(200, identity)
        self.reply(404, {'error': 'unknown route'})

    def do_POST(self):
        started = time.time()
        if self.path != '/tts':
            return self.reply(404, {'error': 'unknown route'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 4096:
                return self.reply(413, {'error': 'body size limit'})
            row = json.loads(self.rfile.read(length))
            text, key, deadline = row['text'], row['id'], float(row['deadline'])
        except (ValueError, KeyError, TypeError):
            return self.reply(400, {'error': 'request schema'})
        if not isinstance(text, str) or not 1 <= len(text) <= 256 or not isinstance(key, str) or not key.isalnum() or len(key) > 40:
            return self.reply(422, {'error': 'character or ID limit'})
        if not math.isfinite(deadline) or not started <= deadline <= started + 60:
            return self.reply(408, {'error': 'deadline'})
        if not owner.acquire(blocking=False):
            return self.reply(429, {'error': 'one owner busy'})
        try:
            audio, receipt = OUTPUT/(key+'.wav'), OUTPUT/(key+'.json')
            if audio.exists() or receipt.exists() or key == 'identity':
                return self.reply(409, {'error': 'duplicate request identity'})
            wave = synthesize(text)
            wavfile.write(audio, model.config.sampling_rate, np.int16(np.clip(wave, -1, 1)*32767))
            ended = time.time()
            result = dict(id=key, text=text, started_epoch=started, ended_epoch=ended, latency_s=ended-started,
                          frames=len(wave), rate=model.config.sampling_rate, finite=True, peak=float(np.abs(wave).max()),
                          sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), execution='GPU', deadline_met=ended<=deadline)
            receipt.write_text(json.dumps(result, indent=2)+'\n')
            self.reply(200 if result['deadline_met'] else 408, result)
        finally:
            owner.release()


if __name__ == '__main__':
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(1606)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this exercise')
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True)
    model = VitsModel.from_pretrained('/model', local_files_only=True).eval().to('cuda')
    synthesize('Ready.')
    identity = dict(torch=torch.__version__, framework_cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                    uid=os.getuid(), model_device=str(next(model.parameters()).device), rate=model.config.sampling_rate,
                    readiness='representative GPU synthesis passed')
    (OUTPUT/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    with ThreadingHTTPServer(('127.0.0.1', 8022), Handler) as server:
        server.daemon_threads = False
        def stop(signum, frame):
            print(json.dumps({'event': 'shutdown_requested', 'signal': signum}), flush=True)
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever()
    print(json.dumps({'event': 'server_drained'}), flush=True)
