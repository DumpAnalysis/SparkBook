"""Explicit image, waveform and scoring contracts; no model downloads."""
from pathlib import Path
import hashlib, math, re
import numpy as np
from PIL import Image, ImageOps
from scipy.io import wavfile
from scipy.signal import resample_poly

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def image_rgb(path, max_pixels=16_000_000):
    with Image.open(path) as im:
        if im.width*im.height > max_pixels: raise ValueError('decoded image exceeds pixel limit')
        im=ImageOps.exif_transpose(im)
        if im.mode in ('RGBA','LA') or (im.mode=='P' and 'transparency' in im.info):
            rgba=im.convert('RGBA');bg=Image.new('RGBA',rgba.size,'white')
            im=Image.alpha_composite(bg,rgba)
        return im.convert('RGB')

def read_audio(path, target_rate=None, max_seconds=120):
    rate,x=wavfile.read(path)
    if not 8000 <= rate <= 192000: raise ValueError('unsupported declared sample rate')
    if x.ndim not in (1,2) or (x.ndim==2 and x.shape[1]>2): raise ValueError('expected mono or stereo WAV')
    if not x.size or len(x)/rate>max_seconds: raise ValueError('empty or overlong audio')
    if x.dtype==np.uint8: x=(x.astype(np.float32)-128)/128
    elif np.issubdtype(x.dtype,np.signedinteger): x=x.astype(np.float32)/float(2**(8*x.dtype.itemsize-1))
    elif np.issubdtype(x.dtype,np.floating): x=x.astype(np.float32)
    else: raise ValueError('unsupported WAV sample type')
    if not np.isfinite(x).all(): raise ValueError('nonfinite waveform')
    if np.max(np.abs(x))>1: raise ValueError('waveform exceeds normalized amplitude range')
    channels=1 if x.ndim==1 else x.shape[1]
    before=dict(source_rate=int(rate),source_frames=len(x),source_channels=channels,source_sha256=digest(path))
    if x.ndim==2: x=x.mean(axis=1)
    if target_rate is not None:
        if not isinstance(target_rate,int) or not 8000<=target_rate<=192000: raise ValueError('invalid target rate')
        if target_rate != rate:
            g=math.gcd(rate,target_rate);x=resample_poly(x,target_rate//g,rate//g).astype(np.float32)
            rate=target_rate
    return rate,x,before

def audio_stats(x,rate):
    x=np.asarray(x)
    if x.ndim not in (1,2) or not x.size or rate<=0 or not np.isfinite(x).all(): raise ValueError('invalid audio stats input')
    # Frames are axis zero, channels are axis one.
    return dict(frames=len(x),channels=1 if x.ndim==1 else x.shape[1],sample_rate=int(rate),
                duration_s=len(x)/rate,peak=float(np.max(np.abs(x))),rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),
                clipped_fraction=float(np.mean(np.abs(x)>=1)))

def write_audio(path,x,rate):
    x=np.asarray(x,dtype=np.float32)
    stats=audio_stats(x,rate)
    if stats['peak']>1: raise ValueError('refusing silent clipping on WAV export')
    wavfile.write(path,rate,np.rint(x*32767).astype(np.int16))
    return stats

def words(text): return re.findall(r'\w+',text.casefold())

def word_error(reference,hypothesis):
    a,b=words(reference),words(hypothesis);d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1): nxt.append(min(nxt[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=nxt
    return dict(reference_words=len(a),hypothesis_words=len(b),errors=d[-1],wer=d[-1]/len(a) if a else None,
                normalization='Unicode word tokens and casefold; punctuation ignored')

def admit(job):
    limits={'vlm':('pixels',1024*1024),'image':('pixels',1024*1024),
            'music':('tokens',512),'tts':('characters',256),'asr':('seconds',30)}
    if job.get('kind') not in limits:return False,'unknown modality'
    field,limit=limits[job['kind']];v=job.get(field)
    if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0:return False,'invalid '+field
    if v>limit:return False,field+' limit'
    return True,'admitted'
