"""Pinned local media inference with bounded inputs and phase records.

No implicit downloads. CUDA runs require the actual laptop. CPU is allowed for
the small TTS/ASR controls, not as an unnoticed fallback from a requested GPU.
"""
import argparse,contextlib,json,os,platform,time,traceback,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from contracts import image_rgb,read_audio,write_audio,audio_stats,digest,word_error
from assets import verify

class Record:
    def __init__(self,a):
        self.a=a;self.events=[];self.start=time.perf_counter();self.result={'status':'running','device':a.device,'seed':a.seed,'phases':self.events}
    def sync(self):
        if self.a.device=='cuda':torch.cuda.synchronize()
    @contextlib.contextmanager
    def stage(self,name):
        self.sync()
        if self.a.device=='cuda':torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            with torch.profiler.record_function('media/'+name):yield
        finally:
            self.sync();row={'name':name,'wall_s':time.perf_counter()-start}
            if self.a.device=='cuda':
                free,total=torch.cuda.mem_get_info();row.update(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),cuda_free=free,cuda_total=total)
            self.events.append(row)
    @contextlib.contextmanager
    def profile(self):
        if not self.a.profile:yield;return
        activities=[torch.profiler.ProfilerActivity.CPU]
        if self.a.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:yield
        prof.export_chrome_trace(str(self.a.out/'trace.json'))
    def finish(self,status,error=None):
        self.result.update(status=status,total_s=time.perf_counter()-self.start)
        if error:self.result['error']=error
        (self.a.out/'result.json').write_text(json.dumps(self.result,indent=2))

def save_wave(a,r,wave,rate):
    wave=np.asarray(wave,dtype=np.float32);np.save(a.out/'waveform.npy',wave)
    r.result['audio']=audio_stats(wave,rate)
    write_audio(a.out/'output.wav',wave,rate)
    infer=next(x['wall_s'] for x in r.events if x['name']=='infer')
    r.result['inference_rtf']=infer/r.result['audio']['duration_s']

def vlm(a,r):
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    with r.stage('load'):
        proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,min_pixels=256*28*28,max_pixels=a.max_pixels,use_fast=False)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype='auto',attn_implementation='sdpa').eval().to(a.device)
        r.result.update(dtype_policy='auto',model_dtype=str(model.dtype),
                        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                        processor_min_pixels=proc.image_processor.min_pixels,
                        processor_max_pixels=proc.image_processor.max_pixels)
    with r.stage('preprocess'):
        image=image_rgb(a.input);image.save(a.out/'normalized.png')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':a.text}]}]
        prompt=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=proc(text=[prompt],images=[image],return_tensors='pt').to(a.device)
        r.result['processed_shapes']={k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)}
        r.result['image_grid_thw']=inputs['image_grid_thw'].tolist()
        r.result['input_tokens']=inputs['input_ids'].shape[1]
    with r.profile(),r.stage('infer'),torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=a.tokens,do_sample=False)
    with r.stage('decode'):
        generated=ids[:,inputs['input_ids'].shape[1]:]
        text=proc.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        r.result.update(text=text,generated_tokens=generated.shape[1]);(a.out/'output.txt').write_text(text+'\n')

def image_generate(a,r):
    from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
    with r.stage('load'):
        cls=StableDiffusionXLImg2ImgPipeline if a.input else StableDiffusionXLPipeline
        pipe=cls.from_pretrained(a.model,local_files_only=True,use_safetensors=True,variant='fp16',torch_dtype=torch.float16)
        if a.tiling:pipe.enable_vae_tiling()
        if a.placement=='resident':pipe.to('cuda')
        elif a.placement=='model-offload':pipe.enable_model_cpu_offload()
        else:pipe.enable_sequential_cpu_offload()
        r.result['scheduler']=type(pipe.scheduler).__name__;r.result['scheduler_config']=dict(pipe.scheduler.config)
    components={}
    for name in ['text_encoder','text_encoder_2','unet','vae']:
        module=getattr(pipe,name)
        components[name]={'parameter_dtypes':sorted({str(p.dtype) for p in module.parameters()}),'parameter_devices':sorted({str(p.device) for p in module.parameters()})}
    r.result['pipeline_components_after_placement']=components
    r.result['vae_force_upcast']=pipe.vae.config.force_upcast
    boundaries={}
    def tensor_meta(x):
        if isinstance(x,torch.Tensor):return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
        if isinstance(x,dict):return {k:tensor_meta(v) for k,v in x.items() if isinstance(v,(torch.Tensor,dict,list,tuple))}
        if isinstance(x,(list,tuple)):return [tensor_meta(v) for v in x]
        return type(x).__name__
    def observe(name,module,args,kwargs):
        if name not in boundaries:
            boundaries[name]={'args':tensor_meta(args),'kwargs':tensor_meta(kwargs),'parameter_dtypes':sorted({str(p.dtype) for p in module.parameters()})}
            (a.out/'pipeline_boundaries.json').write_text(json.dumps(boundaries,indent=2))
    if a.trace_steps:
        for name,module in [('text_encoder',pipe.text_encoder),('text_encoder_2',pipe.text_encoder_2),('unet',pipe.unet),('vae_decoder',pipe.vae.decoder)]:
            module.register_forward_pre_hook(lambda m,args,kwargs,name=name:observe(name,m,args,kwargs),with_kwargs=True)
    steps=[]
    def callback(pipeline,i,t,kwargs):
        z=kwargs['latents'];finite=bool(torch.isfinite(z).all())
        steps.append({'step':i,'timestep':float(t),'rms':float(z.float().square().mean().sqrt()),'finite':finite,'shape':list(z.shape),'dtype':str(z.dtype),'device':str(z.device)})
        if not finite:raise FloatingPointError('nonfinite diffusion latent')
        return kwargs
    kw=dict(prompt=a.text,num_inference_steps=a.steps,guidance_scale=a.guidance,generator=torch.Generator('cpu').manual_seed(a.seed))
    if a.input:
        with r.stage('preprocess'):kw.update(image=image_rgb(a.input).resize((a.width,a.height)),strength=a.strength)
    else:kw.update(height=a.height,width=a.width)
    if a.trace_steps:kw.update(callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
    try:
        with r.profile(),r.stage('infer'),torch.inference_mode():result=pipe(**kw).images[0]
        with r.stage('encode'):result.save(a.out/'output.png')
    finally:(a.out/'denoising_steps.json').write_text(json.dumps(steps,indent=2))
    r.result.update(output_dimensions=list(result.size),placement=a.placement,instrumented_steps=a.trace_steps)

def music(a,r):
    from transformers import AutoProcessor,MusicgenForConditionalGeneration,MusicgenConfig
    class PinnedMusicgen(MusicgenForConditionalGeneration):
        # Transformers4.57.1 infers decoder config from the shared base class.
        config_class=MusicgenConfig
    with r.stage('load'):
        proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
        model=PinnedMusicgen.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype=torch.float32).eval().to(a.device)
    with r.stage('preprocess'):inputs=proc(text=[a.text],padding=True,return_tensors='pt').to(a.device)
    with r.profile(),r.stage('infer'),torch.inference_mode():
        values=model.generate(**inputs,do_sample=True,guidance_scale=a.guidance,max_new_tokens=a.tokens)
    with r.stage('encode'):
        # HF returns batch, channels, samples; WAV expects frames, channels.
        w=values[0].float().cpu().numpy().T
        if w.shape[1]==1:w=w[:,0]
        save_wave(a,r,w,model.config.audio_encoder.sampling_rate)
    r.result['requested_new_tokens']=a.tokens

def tts(a,r):
    from transformers import AutoTokenizer,VitsModel
    with r.stage('load'):
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        model=VitsModel.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype=torch.float32).eval().to(a.device)
    with r.stage('preprocess'):inputs=tok(a.text,return_tensors='pt').to(a.device)
    with r.profile(),r.stage('infer'),torch.inference_mode():w=model(**inputs).waveform[0].float().cpu().numpy()
    with r.stage('encode'):save_wave(a,r,w,model.config.sampling_rate)

def asr(a,r):
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    with r.stage('load'):
        proc=WhisperProcessor.from_pretrained(a.model,local_files_only=True)
        model=WhisperForConditionalGeneration.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype=torch.float32).eval().to(a.device)
    with r.stage('preprocess'):
        rate,w,source=read_audio(a.input,16000,max_seconds=30)
        inputs=proc(w,sampling_rate=rate,return_tensors='pt',return_attention_mask=True).to(a.device)
        r.result.update(audio=audio_stats(w,rate),source=source,feature_shape=list(inputs['input_features'].shape))
    with r.profile(),r.stage('infer'),torch.inference_mode():
        ids=model.generate(**inputs,language='en',task='transcribe',max_new_tokens=a.tokens,do_sample=False)
    with r.stage('decode'):
        text=proc.batch_decode(ids,skip_special_tokens=True)[0];r.result['text']=text
        (a.out/'output.txt').write_text(text+'\n')
        if a.reference:r.result['score']=word_error(a.reference.read_text(),text)
    r.result['inference_rtf']=next(x['wall_s'] for x in r.events if x['name']=='infer')/r.result['audio']['duration_s']

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['vlm','image','music','tts','asr']);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--input',type=Path);p.add_argument('--text',default='The graphics processor has thirty gigabytes available.')
    p.add_argument('--reference',type=Path);p.add_argument('--seed',type=int,default=1606);p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--max-pixels',type=int,default=512*28*28);p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=512);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--guidance',type=float,default=3.0);p.add_argument('--strength',type=float,default=.5)
    p.add_argument('--placement',choices=['resident','model-offload','sequential-offload'],default='resident');p.add_argument('--tiling',action='store_true');p.add_argument('--trace-steps',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    if a.kind in ('vlm','asr') and not a.input:p.error('this modality needs --input')
    if a.device=='cpu' and a.kind not in ('tts','asr'):p.error('CPU target controls are limited to tts and asr')
    if a.kind=='tts' and not 1<=len(a.text)<=256:p.error('TTS text limit is 256 characters')
    if not 1<=a.tokens<=512:p.error('token budget must be 1..512')
    if not 256*28*28<=a.max_pixels<=1024*28*28:p.error('vision pixel budget out of bounds')
    if any(v<256 or v>1024 or v%8 for v in (a.width,a.height)):p.error('image dimensions must be multiples of eight in 256..1024')
    if not 1<=a.steps<=50 or not .1<=a.strength<=1:p.error('invalid diffusion steps or strength')
    if not np.isfinite(a.guidance) or not 1<=a.guidance<=12:p.error('invalid guidance')
    if a.snapshot and a.device!='cuda':p.error('allocator snapshots require CUDA')
    a.out.mkdir(parents=True,exist_ok=False);r=Record(a);history=False
    try:
        if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        key={'vlm':'qwen-vl','image':'sdxl','music':'musicgen','tts':'mms','asr':'whisper'}[a.kind]
        with r.stage('verify'):
            spec=verify(a.model,key)
            if a.input:r.result['input_sha256']=digest(a.input)
        env=dict(python=platform.python_version(),machine=platform.machine(),torch=torch.__version__,torch_cuda=torch.version.cuda,
                 packages={n:importlib.metadata.version(n) for n in ['transformers','numpy','scipy','Pillow','safetensors']})
        if a.device=='cuda':env.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
        r.result.update(model=spec,environment=env,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
        torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
        if a.snapshot:
            torch.cuda.memory._record_memory_history(max_entries=100000);history=True
        {'vlm':vlm,'image':image_generate,'music':music,'tts':tts,'asr':asr}[a.kind](a,r)
        r.result['output_sha256']={p.name:digest(p) for p in a.out.iterdir() if p.is_file()}
        r.finish('ok');print(json.dumps({'status':'ok','kind':a.kind,'device':a.device,'result':str(a.out/'result.json')},sort_keys=True))
        if 'text' in r.result:print(r.result['text'])
        if 'audio' in r.result:print(json.dumps(r.result['audio'],sort_keys=True))
        if 'score' in r.result:print(json.dumps(r.result['score'],sort_keys=True))
    except Exception as e:
        r.finish('failed',f'{type(e).__name__}: {e}');traceback.print_exc();raise SystemExit(1)
    finally:
        if history:
            try:torch.cuda.memory._dump_snapshot(str(a.out/'allocator.pickle'))
            finally:torch.cuda.memory._record_memory_history(enabled=None)
if __name__=='__main__':main()
