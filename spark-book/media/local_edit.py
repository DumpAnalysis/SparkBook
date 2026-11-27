"""Create a hue guide inside a reader-selected mask, then preserve the source outside it."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    guide=sub.add_parser('guide')
    area=guide.add_mutually_exclusive_group(required=True)
    area.add_argument('--polygon',metavar='X,Y',nargs='+')
    area.add_argument('--mask',type=Path)
    guide.add_argument('--hue',type=int,default=85,help='Pillow hue, 0..255')
    finish=sub.add_parser('compose')
    finish.add_argument('--mask',type=Path,required=True)
    finish.add_argument('--generated',type=Path,required=True)
    for command in (guide,finish):
        command.add_argument('--source',type=Path,required=True)
        command.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    source=Image.open(a.source).convert('RGB')
    if a.mask:
        mask=Image.open(a.mask).convert('L')
        if mask.size!=source.size:p.error('mask and source dimensions differ')
    else:
        try:points=[tuple(map(int,point.split(','))) for point in a.polygon]
        except ValueError:p.error('expected polygon vertices in x,y format')
        if len(points)<3 or any(len(pt)!=2 or not (0<=pt[0]<source.width and 0<=pt[1]<source.height) for pt in points):
            p.error('at least three vertices inside the source are required')
        mask=Image.new('L',source.size);ImageDraw.Draw(mask).polygon(points,fill=255)
    if mask.getextrema()!=(0,255):p.error('mask must include a selected and an unselected region')
    a.out.mkdir(parents=True,exist_ok=False)
    if a.command=='guide':
        if not 0<=a.hue<=255:p.error('hue must be 0..255')
        hue,sat,value=source.convert('HSV').split()
        target=Image.merge('HSV',(Image.new('L',source.size,a.hue),sat,value)).convert('RGB')
        Image.composite(target,source,mask).save(a.out/'guide.png')
        mask.save(a.out/'mask.png')
    else:
        generated=Image.open(a.generated).convert('RGB')
        if generated.size!=source.size:p.error('generated and source dimensions differ')
        result=Image.composite(generated,source,mask)
        result.save(a.out/'output.png')
        inside=np.asarray(mask)>0
        changed=np.any(np.asarray(result)!=np.asarray(source),axis=2)
        stats={'mask_pixels':int(inside.sum()),'changed_inside':int(changed[inside].sum()),
               'changed_outside':int(changed[~inside].sum())}
        (a.out/'comparison.json').write_text(json.dumps(stats,indent=2)+'\n')
        print(stats)


if __name__=='__main__':
    main()
