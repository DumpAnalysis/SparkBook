"""Compare one locally trained SDXL LoRA with its unchanged base prompt and seed."""
import argparse,json
from pathlib import Path
import torch
from assets import verify
from contracts import digest
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--text',default='A green square on a white background');a=p.parse_args()
    verify(a.model,'sdxl')
    if not torch.cuda.is_available():raise RuntimeError('CUDA target required')
    from diffusers import StableDiffusionXLPipeline
    a.out.mkdir(parents=True,exist_ok=False)
    pipe=StableDiffusionXLPipeline.from_pretrained(a.model,local_files_only=True,use_safetensors=True,variant='fp16',torch_dtype=torch.float16)
    pipe.load_lora_weights(a.adapter,adapter_name='lab',weight_name='pytorch_lora_weights.safetensors',local_files_only=True);pipe.set_adapters('lab',adapter_weights=1.0)
    pipe.enable_model_cpu_offload()
    files={f.name:digest(f) for f in a.adapter.glob('*.safetensors')}
    if not files:raise ValueError('no adapter safetensors found')
    for enabled,name in [(False,'base'),(True,'adapter')]:
        if enabled:pipe.enable_lora()
        else:pipe.disable_lora()
        with torch.no_grad():
            image=pipe(a.text,width=512,height=512,num_inference_steps=20,guidance_scale=5,generator=torch.Generator('cpu').manual_seed(1610)).images[0]
        image.save(a.out/(name+'.png'))
    (a.out/'comparison.json').write_text(json.dumps(dict(prompt=a.text,seed=1610,scale=1.0,adapter_files=files),indent=2));print('saved base.png and adapter.png')
if __name__=='__main__':main()
