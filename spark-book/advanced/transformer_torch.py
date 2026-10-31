"""PyTorch translation of the portable single-head transformer reference."""
import argparse
import math
import torch
from reference import inputs, full_block, save_new


def run(device):
    dtype = torch.float64  # diagnostic reference precision, not production speed
    x = torch.tensor(inputs(), device=device, dtype=dtype)
    def proj(v, width, salt):
        i = torch.arange(v.shape[-1], device=device)[:, None]
        j = torch.arange(width, device=device)[None, :]
        w = ((i * 7 + j * 3 + salt) % 11 - 5).to(dtype) / 20
        return v @ w
    def rms(v):
        return v / torch.sqrt(v.square().mean(-1, keepdim=True) + 1e-5)
    def rope(v, pos):
        freq = 10000 ** (torch.arange(0, 8, 2, device=device, dtype=dtype) / 8)
        theta = torch.as_tensor(pos, device=device, dtype=dtype)[..., None] / freq
        a, b = v[..., 0::2], v[..., 1::2]
        return torch.stack((a * theta.cos() - b * theta.sin(),
                            a * theta.sin() + b * theta.cos()), -1).flatten(-2)
    def finish(v, attention):
        r = v + proj(attention, 8, 4)
        h = torch.nn.functional.silu(proj(rms(r), 16, 5))
        return r + proj(h, 8, 6)
    z = rms(x)
    q = rope(proj(z, 8, 1), torch.arange(5, device=device))
    k = rope(proj(z, 8, 2), torch.arange(5, device=device))
    v = proj(z, 8, 3)
    scores = q @ k.T / math.sqrt(8)
    mask = torch.ones(5, 5, device=device, dtype=torch.bool).triu(1)
    probs = scores.masked_fill(mask, -float('inf')).softmax(-1)
    full = finish(x, probs @ v)
    keys, values, outs = [], [], []
    for t in range(5):
        zi = rms(x[t])
        qi = rope(proj(zi, 8, 1), t)
        keys.append(rope(proj(zi, 8, 2), t))
        values.append(proj(zi, 8, 3))
        p = (qi @ torch.stack(keys).T / math.sqrt(8)).softmax(-1)
        outs.append(finish(x[t], p @ torch.stack(values)))
    inc = torch.stack(outs)
    ref = torch.tensor([r['output'] for r in full_block(inputs())],
                       device=device, dtype=dtype)
    torch.testing.assert_close(full, ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(inc, ref, atol=1e-10, rtol=1e-10)
    return {'device': str(device), 'dtype': str(dtype), 'q': q.cpu().tolist(),
            'k': k.cpu().tolist(), 'v': v.cpu().tolist(),
            'probabilities': probs.cpu().tolist(), 'output': full.cpu().tolist(),
            'incremental': inc.cpu().tolist(), 'reference_pass': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with torch.no_grad():
        report = run(a.device)
    save_new(a.out, report)
    print('device=' + a.device + ' dtype=float64 full_incremental_reference=PASS')
    print('saved=' + a.out)


if __name__ == '__main__':
    main()
