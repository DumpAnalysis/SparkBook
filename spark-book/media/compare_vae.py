"""Check an SDXL VAE tiling treatment against matched full-frame controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(directory):
    receipt = json.loads((directory / 'result.json').read_text())
    png = directory / 'output.png'
    if receipt['status'] != 'ok' or hashlib.sha256(png.read_bytes()).hexdigest() != receipt['output_sha256']['output.png']:
        raise ValueError(f'Incomplete or mismatched image in {directory}')
    with Image.open(png) as im:
        pixels = np.array(im.convert('RGB'), dtype=np.int16)
    return receipt, pixels


def compare(directories):
    rows = [load(d) for d in directories]
    plain, default, tiled = [r['vae_decode'] for r, _ in rows]
    keys = ('text', 'seed', 'width', 'height', 'steps', 'guidance', 'placement', 'input', 'strength')
    def identity(receipt):
        config = dict(receipt['scheduler_config'])
        # Diffusers serializes this set of field names in process-dependent order.
        if '_use_default_values' in config:
            config['_use_default_values'] = sorted(config['_use_default_values'])
        return [receipt['model'], receipt['scheduler'], config,
                *[receipt['arguments'][k] for k in keys], receipt['vae_decode']['latent_sha256'],
                receipt['vae_decode']['latent_shape']]
    if any(identity(r) != identity(rows[0][0]) for r, _ in rows[1:]):
        raise ValueError('Model, generation settings or final latent differ between runs')
    full = plain['latent_shape'][-2:]
    if plain['enabled'] or not default['enabled'] or not tiled['enabled']:
        raise ValueError('Expected disabled, default-enabled and reduced-tile treatments')
    if plain['decoder_inputs'] != [full] or default['decoder_inputs'] != [full]:
        raise ValueError('Both full-frame controls must actually decode the complete latent once')
    tiles = tiled['decoder_inputs']
    if not 0 < tiled['tile_latents'] < min(full) or len(tiles) < 2 or any(
        len(shape) != 2 or min(shape) <= 0 or max(shape) > tiled['tile_latents'] for shape in tiles
    ):
        raise ValueError('Reduced tile size did not produce multiple bounded decoder calls')
    result = dict(latent_sha256=plain['latent_sha256'], latent_shape=plain['latent_shape'], runs={})
    for name, directory, (receipt, pixels) in zip(('plain', 'default', 'tiled'), directories, rows, strict=True):
        delta = np.abs(pixels - rows[0][1])
        infer = next(p for p in receipt['phases'] if p['name'] == 'infer')
        result['runs'][name] = dict(source=str(directory), **receipt['vae_decode'],
                                    peak_infer_allocated_bytes=infer['peak_allocated'], mean_abs_rgb=float(delta.mean()),
                                    max_abs_rgb=int(delta.max()), changed_pixels=int(np.any(delta, axis=2).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plain', 'default', 'tiled', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    report = compare([args.plain, args.default, args.tiled])
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print('Decoder calls', {name: len(r['decoder_inputs']) for name, r in report['runs'].items()})


if __name__ == '__main__':
    main()
