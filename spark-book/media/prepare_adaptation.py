"""Create two small teaching manifests; input WAVs must be distinct recordings."""
import argparse,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from contracts import digest
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--train-audio',type=Path);p.add_argument('--dev-audio',type=Path);a=p.parse_args()
    if bool(a.train_audio)!=bool(a.dev_audio):p.error('supply both WAV inputs or neither')
    if a.train_audio and digest(a.train_audio)==digest(a.dev_audio):p.error('train and dev audio must differ')
    a.out.mkdir(parents=True,exist_ok=False)
    if a.train_audio:
        rows=[]
        for split,src,text in [('train',a.train_audio,'The processor is ready.'),('dev',a.dev_audio,'Retain the complete diagnostic record.')]:
            shutil.copy2(src,a.out/(split+'.wav'));rows.append(dict(audio=split+'.wav',text=text,split=split))
        (a.out/'asr.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print('ASR teaching rows 2; verify their transcripts')
    else:
        rows=[]
        for i,(color,name) in enumerate([((220,30,30),'red'),((30,60,220),'blue')]):
            im=Image.new('RGB',(512,512),'white');ImageDraw.Draw(im).rectangle((130,130,382,382),fill=color)
            f=f'{i:03d}.png';im.save(a.out/f);rows.append(dict(file_name=f,text='A '+name+' square on a white background'))
        (a.out/'metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print('Image plumbing examples 2; not a quality corpus')
if __name__=='__main__':main()
