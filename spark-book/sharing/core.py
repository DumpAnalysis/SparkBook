"""Offline embedding fixture. No downloaded model or semantic quality claim."""
import json
from pathlib import Path
import numpy as np


def weights():
    e = (np.sin(np.arange(256 * 64, dtype=np.float64) * .013)
         .reshape(256, 64) * .1).astype('float32')
    w = (np.eye(64) + np.cos(np.arange(4096).reshape(64, 64) * .017)
         * .003).astype('float32')
    b = np.linspace(-.01, .01, 64, dtype='float32')
    return e, w, b


def tokens(request_id):
    # Fixed integer tokens keep results independent of tokenizer downloads.
    return [4 + (request_id * 11 + j * 7) % 252 for j in range(16)]


def oracle(ids):
    e, w, b = weights()
    y = np.tanh(e[np.asarray(ids)].mean(axis=1) @ w.T + b)
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def model(device):
    import torch
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 64)
            self.projection = torch.nn.Linear(64, 64)
            e, w, b = weights()
            with torch.no_grad():
                self.embedding.weight.copy_(torch.from_numpy(e))
                self.projection.weight.copy_(torch.from_numpy(w))
                self.projection.bias.copy_(torch.from_numpy(b))
        def forward(self, x):
            y = self.projection(self.embedding(x).mean(dim=1)).tanh()
            return torch.nn.functional.normalize(y, dim=1)
    torch.set_num_threads(1)
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    return Encoder().to(device).eval()


def memory(device):
    import torch
    if device != 'cuda':
        return {'allocated': None, 'reserved': None,
                'peak_allocated': None, 'peak_reserved': None}
    return dict(allocated=torch.cuda.memory_allocated(0),
                reserved=torch.cuda.memory_reserved(0),
                peak_allocated=torch.cuda.max_memory_allocated(0),
                peak_reserved=torch.cuda.max_memory_reserved(0))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
