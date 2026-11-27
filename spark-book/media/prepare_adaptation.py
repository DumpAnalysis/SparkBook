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
        # Withhold every green square. Green occurs on circles, and square
        # geometry occurs in other colors at several sizes and positions.
        palette={'red':(210,35,35),'blue':(35,65,210),'green':(30,155,50),'orange':(225,125,25)}
        for name,color in palette.items():
            for shape in ('circle','square'):
                if (name,shape)==('green','square'):
                    continue
                for side,dx,dy in [(250,0,0),(300,-22,12),(210,18,-18)]:
                    x=(512-side)//2+dx; y=(512-side)//2+dy
                    im=Image.new('RGB',(512,512),'white'); draw=ImageDraw.Draw(im)
                    paint=draw.ellipse if shape=='circle' else draw.rectangle
                    paint((x,y,x+side-1,y+side-1),fill=color)
                    filename=f'{name}-{shape}-{side}.png';im.save(a.out/filename)
                    rows.append(dict(file_name=filename,text=f'A {name} {shape} on a white background'))
        (a.out/'metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(f'{len(rows)} training images; held out all green squares')

if __name__=='__main__':main()
