"""Train a tiny 2D epsilon predictor; not a pretrained text-to-image pipeline."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn

def schedule(steps=100):
    if steps<2:raise ValueError('need at least two steps')
    beta=torch.linspace(.0001,.08,steps);alpha=1-beta
    return beta,alpha,torch.cumprod(alpha,0)

def mix(x,e,t,abar):return abar[t,None].sqrt()*x+(1-abar[t,None]).sqrt()*e
def reconstruct(xt,e,t,abar):return (xt-(1-abar[t,None]).sqrt()*e)/abar[t,None].sqrt()

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x,t):
        angle=t.float()[:,None]/99*math.pi
        return self.net(torch.cat([x,torch.sin(angle),torch.cos(angle)],1))

def points(n,g):
    k=torch.randint(8,(n,),generator=g);ang=k*math.pi/4
    return 2*torch.stack([torch.cos(ang),torch.sin(ang)],1)+.08*torch.randn(n,2,generator=g)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
    if not 1<=a.steps<=10000:p.error('training steps must be 1..10000')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1603)
    g=torch.Generator().manual_seed(1603);b,al,ab=schedule();model=Denoiser();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    testg=torch.Generator().manual_seed(1604);x=points(512,testg);e=torch.randn(512,2,generator=testg);t=torch.randint(100,(512,),generator=testg);xt=mix(x,e,t,ab)
    with torch.no_grad():initial=float(((model(xt,t)-e)**2).mean())
    oracle=float((reconstruct(xt,e,t,ab)-x).abs().max());wrong=float(((reconstruct(xt,-e,t,ab)-x)**2).mean())
    losses=[]
    for step in range(a.steps):
        clean=points(128,g);noise=torch.randn(128,2,generator=g);ts=torch.randint(100,(128,),generator=g)
        loss=((model(mix(clean,noise,ts,ab),ts)-noise)**2).mean();opt.zero_grad();loss.backward();opt.step()
        if (step+1)%500==0 or step==0:losses.append([step+1,float(loss.detach())])
    with torch.no_grad():
        final=float(((model(xt,t)-e)**2).mean());sample=torch.randn(512,2,generator=g)
        for k in reversed(range(100)):
            ts=torch.full((512,),k,dtype=torch.long);eps=model(sample,ts)
            mean=(sample-b[k]/(1-ab[k]).sqrt()*eps)/al[k].sqrt()
            variance=b[k]*(1-ab[k-1])/(1-ab[k]) if k else torch.tensor(0.)
            sample=mean+variance.sqrt()*torch.randn(sample.shape,generator=g) if k else mean
    assert oracle<1e-5 and wrong>1 and final<initial
    torch.save({'state_dict':model.state_dict(),'steps':a.steps,'seed':1603,'beta':b},a.out/'denoiser.pt')
    np.save(a.out/'samples.npy',sample.numpy());np.save(a.out/'reference.npy',x.numpy())
    result=dict(device='cpu',parameters=sum(p.numel() for p in model.parameters()),training_steps=a.steps,
                initial_held_noise_mse=initial,final_held_noise_mse=final,oracle_max_error=oracle,wrong_noise_mse=wrong,
                sample_count=len(sample),finite_samples=bool(torch.isfinite(sample).all()))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));(a.out/'losses.json').write_text(json.dumps(losses))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7,3.2))
    for ax,data,title in [(axes[0],x,'Held-out distribution'),(axes[1],sample,'Tiny diffusion samples')]:
        ax.scatter(data[:,0],data[:,1],s=5,alpha=.55);ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='x',ylabel='y',title=title);ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(a.out/'diffusion.png',dpi=180);plt.close(fig)
    print(json.dumps(result,sort_keys=True))
    for step,value in losses:print(f'step={step} noise_mse={value:.6f}')
if __name__=='__main__':main()
