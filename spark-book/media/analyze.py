"""Inspect a WAV or compare two texts; preserve data rather than playing audio."""
import argparse,json
from pathlib import Path
import numpy as np
from contracts import read_audio,audio_stats,word_error,write_audio
def main():
    p=argparse.ArgumentParser();p.add_argument('--audio',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--hypothesis',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--rate',type=int);a=p.parse_args()
    if not a.audio and not (a.reference and a.hypothesis):p.error('choose audio or both text paths')
    a.out.mkdir(parents=True,exist_ok=False)
    if a.audio:
        rate,x,source=read_audio(a.audio,a.rate);report=dict(source=source,mono=audio_stats(x,rate))
        if a.rate:write_audio(a.out/'resampled.wav',x,rate)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import stft
        f,t,z=stft(x,rate,nperseg=min(512,len(x)))
        fig,ax=plt.subplots(2,1,figsize=(7,4.5),layout='constrained')
        ax[0].plot(np.arange(len(x))/rate,x,lw=.4);ax[0].set(xlabel='Time (s)',ylabel='Amplitude',title='Decoded mono waveform')
        q=ax[1].pcolormesh(t,f/1000,20*np.log10(np.maximum(abs(z),1e-8)),shading='auto',vmin=-100,vmax=0)
        ax[1].set(xlabel='Time (s)',ylabel='Frequency (kHz)',title='STFT magnitude');fig.colorbar(q,ax=ax[1],label='dB relative to amplitude one')
        fig.savefig(a.out/'audio.png',dpi=180);plt.close(fig)
    else:report=word_error(a.reference.read_text(),a.hypothesis.read_text())
    (a.out/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
