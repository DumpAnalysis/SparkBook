"""Measure warm-model time until the local output receives its first playable PCM sample."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel
from assets import verify
from contracts import write_audio


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--phrase',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not 1<=len(a.phrase)<=8 or any(not 1<=len(s.strip())<=256 for s in a.phrase):
        p.error('supply 1..8 phrases, each 1..256 characters')
    verify(a.model,'mms')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=VitsModel.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype=torch.float32).to('cuda').eval()
    rate=model.config.sampling_rate
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1606);torch.cuda.synchronize()
    start=time.perf_counter();events=[];chunks=[];first=None
    with (a.out/'stream.s16le').open('xb',buffering=0) as sink, torch.inference_mode():
        for text in a.phrase:
            inputs=tokenizer(text,return_tensors='pt').to('cuda')
            wave=model(**inputs).waveform[0].float().cpu().numpy()
            if not wave.size or not np.isfinite(wave).all() or np.abs(wave).max()>1:
                raise ValueError('invalid waveform')
            pcm=np.rint(wave*32767).astype('<i2').tobytes()
            if first is None:
                sink.write(pcm[:2])
                first=time.perf_counter()-start
                pcm=pcm[2:]
            sink.write(pcm)
            events.append(dict(phrase=text,frames=len(wave),delivered_s=time.perf_counter()-start))
            chunks.append(wave)
    write_audio(a.out/'output.wav',np.concatenate(chunks),rate)
    record=dict(sample_rate=rate,first_pcm_s=first,phrases=events,frames=sum(len(w) for w in chunks),
                scope='Warm model to unbuffered local PCM sink; synthesis returns a complete phrase at a time.')
    (a.out/'timing.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
