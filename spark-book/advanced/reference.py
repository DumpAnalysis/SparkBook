"""Portable numerical oracles. No CUDA or PyTorch execution."""
import argparse
import json
import math
from pathlib import Path


def save_new(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def quantize(values, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    if not all(math.isfinite(x) for x in values):
        raise ValueError('values must be finite')
    codes = []
    for x in values:
        u = max(-8.0, min(7.0, x / scale))
        q = math.floor(u + .5) if u >= 0 else math.ceil(u - .5)
        codes.append(q)
    packed = []
    for i in range(0, len(codes), 2):
        hi = codes[i + 1] & 15 if i + 1 < len(codes) else 0
        packed.append((codes[i] & 15) | (hi << 4))
    return codes, packed


def dequantize(packed, n, scale):
    if n < 0 or len(packed) != (n + 1) // 2:
        raise ValueError('packed length does not match element count')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    if any(type(b) is not int or not 0 <= b < 256 for b in packed):
        raise ValueError('packed byte outside 0..255')
    out = []
    for i in range(n):
        q = (packed[i // 2] >> (4 * (i % 2))) & 15
        out.append((q - 16 if q >= 8 else q) * scale)
    return out


def projection(x, nout, salt):
    # Deterministic non-symmetric weights, [input, output].
    return [sum(v * (((i * 7 + j * 3 + salt) % 11) - 5) / 20
                for i, v in enumerate(x)) for j in range(nout)]


def rms(x):
    denom = math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
    return [v / denom for v in x]


def rope(x, pos):
    if len(x) % 2:
        raise ValueError('RoPE needs even width')
    out = []
    for i in range(0, len(x), 2):
        angle = pos / (10000 ** (i / len(x)))
        c, s = math.cos(angle), math.sin(angle)
        out.extend((x[i] * c - x[i + 1] * s,
                    x[i] * s + x[i + 1] * c))
    return out


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    return [v / sum(e) for v in e]


def inputs():
    return [[((t * 5 + i * 3) % 13 - 6) / 7
             for i in range(8)] for t in range(5)]


def finish(x, q, keys, values):
    scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q))
              for k in keys]
    probs = softmax(scores)
    attn = [sum(p * v[j] for p, v in zip(probs, values))
            for j in range(len(x))]
    residual = [a + b for a, b in zip(x, projection(attn, 8, 4))]
    hidden = projection(rms(residual), 16, 5)
    hidden = [v / (1 + math.exp(-v)) for v in hidden]  # SiLU
    out = [a + b for a, b in zip(residual, projection(hidden, 8, 6))]
    return {'q': q, 'scores': scores, 'probabilities': probs,
            'attention': attn, 'residual': residual, 'output': out}


def full_block(xs):
    normalized = [rms(x) for x in xs]
    qs = [rope(projection(x, 8, 1), i) for i, x in enumerate(normalized)]
    ks = [rope(projection(x, 8, 2), i) for i, x in enumerate(normalized)]
    vs = [projection(x, 8, 3) for x in normalized]
    return [finish(x, qs[t], ks[:t + 1], vs[:t + 1])
            for t, x in enumerate(xs)]


def incremental_block(xs, wrong_position=False):
    keys, values, rows = [], [], []
    for t, x in enumerate(xs):
        z = rms(x)
        pos = 0 if wrong_position else t
        q = rope(projection(z, 8, 1), pos)
        keys.append(rope(projection(z, 8, 2), pos))
        values.append(projection(z, 8, 3))
        rows.append(finish(x, q, keys, values))
    return rows


def max_error(a, b, key='output'):
    return max(abs(x - y) for ra, rb in zip(a, b)
               for x, y in zip(ra[key], rb[key]))


def transformer_report():
    xs = inputs()
    full = full_block(xs)
    good = incremental_block(xs)
    bad = incremental_block(xs, True)
    assert max_error(full, good) < 1e-12
    assert max_error(full, bad) > 1e-5
    first = next(i for i, (a, b) in enumerate(zip(full, bad))
                 if any(abs(x - y) > 1e-12 for x, y in zip(a['q'], b['q'])))
    return {'evidence': 'CPU reference; no model weights or GPU execution',
            'shape': [5, 8], 'cache_elements': 2 * 5 * 8,
            'good_max_error': max_error(full, good),
            'bad_max_error': max_error(full, bad),
            'first_bad_q_token': first,
            'full': full, 'incremental': good, 'wrong_position': bad}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['transformer', 'int4'])
    p.add_argument('--out')
    a = p.parse_args()
    if a.mode == 'transformer':
        r = transformer_report()
        print('evidence=CPU reference; no GPU execution')
        print('shape=5x8 cache_elements=80')
        print(f"full_vs_incremental_max_error={r['good_max_error']:.12f}")
        print(f"wrong_position_max_error={r['bad_max_error']:.6f}")
        print(f"first_bad_q_token={r['first_bad_q_token']}")
        print('causal_prefix_and_repair=PASS')
    else:
        x = [-3., -2., -.375, -.125, 0., .125, .375, 1.75, 3.]
        q, b = quantize(x, .25)
        y = dequantize(b, len(x), .25)
        r = {'evidence': 'CPU INT4 teaching format', 'input': x,
             'scale': .25, 'codes': q, 'packed': b, 'decoded': y}
        print('evidence=CPU reference; no GPU execution')
        print('codes=' + ','.join(map(str, q)))
        print('packed_hex=' + ' '.join(f'{v:02x}' for v in b))
        print('decoded=' + ','.join(f'{v:g}' for v in y))
        print('tail_high_nibble=0 round_ties=away_from_zero')
    if a.out:
        save_new(a.out, r)
        print('saved=' + a.out)


if __name__ == '__main__':
    main()
