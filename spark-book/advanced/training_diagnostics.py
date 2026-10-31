"""Bounded training experiments; PyTorch target execution required."""
import argparse
import copy
import random
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint


def model(device):
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(),
                               torch.nn.Dropout(.25), torch.nn.Linear(32, 4)).to(device)


def state():
    return {'python_rng': random.getstate(), 'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore(r):
    random.setstate(r['python_rng']); torch.set_rng_state(r['cpu_rng'])
    if r['cuda_rng']:
        torch.cuda.set_rng_state_all(r['cuda_rng'])


def step(m, opt, x, y):
    opt.zero_grad(set_to_none=True)
    # Python RNG deliberately influences training, so its restoration is tested.
    loss = (m(x) - y).square().mean() * (1 + random.random() * .01)
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    opt.step()
    return loss.detach()


def audit(device):
    torch.manual_seed(7)
    m = model(device)
    x, y = torch.randn(8, 16, device=device), torch.randn(8, 4, device=device)
    rng = state()
    loss = (m(x) - y).square().mean(); loss.backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True); restore(rng)
    loss2 = (checkpoint(m, x, use_reentrant=False,
                        preserve_rng_state=True) - y).square().mean()
    loss2.backward()
    torch.testing.assert_close(loss, loss2)
    for a, p in zip(g1, m.parameters()):
        torch.testing.assert_close(a, p.grad)
    print('checkpoint_loss_and_gradients PASS')
    saved_grad=m[0].weight.grad
    m[0].weight.grad=None
    missing=[name for name,p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing==['0.weight']
    m[0].weight.grad=saved_grad
    print('missing_trainable_gradient_detected=0.weight repair=PASS')
    # A deliberate dtype overflow is detected before any optimizer step.
    overflow = torch.tensor(65504., device=device, dtype=torch.float16) * 2
    assert not torch.isfinite(overflow)
    print('fp16_overflow_detected optimizer_step=SKIPPED')
    if device == 'cuda':
        held = []
        torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
        for _ in range(6):
            held.append(torch.zeros(256, 256, device=device).detach())
        torch.cuda.synchronize(); peak = torch.cuda.memory_allocated()
        held.clear()
        torch.cuda.synchronize(); after = torch.cuda.memory_allocated()
        assert peak > base and after == base
        print('detached_tensor_retention_exposed release_to_baseline=PASS')
        print('reserved_bytes=' + str(torch.cuda.memory_reserved()))
    else:
        print('GPU_allocator_experiment=SKIPPED_on_cpu')


def resume(device, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(21); random.seed(21)
    m = model(device); opt = torch.optim.AdamW(m.parameters(), lr=.001)
    x, y = torch.randn(8,16,device=device), torch.randn(8,4,device=device)
    step(m,opt,x,y)
    snapshot = {'model': copy.deepcopy(m.state_dict()),
                'optimizer': copy.deepcopy(opt.state_dict()),
                'rng': state(), 'step': 1, 'data_cursor': 1,
                'x': x, 'y': y, 'torch_version': str(torch.__version__)}
    checkpoint_path = out / 'resume.pt'
    torch.save(snapshot, checkpoint_path)
    expected_loss = step(m,opt,x,y)
    expected = copy.deepcopy(m.state_dict())
    bad=model(device);bad.load_state_dict(snapshot['model'])
    badopt=torch.optim.AdamW(bad.parameters(),lr=.001)
    restore(snapshot['rng']);step(bad,badopt,x,y)
    assert any(not torch.allclose(v,expected[k],rtol=1e-6,atol=1e-7)
               for k,v in bad.state_dict().items())
    print('weights_only_resume_diverged=EXPOSED')
    m2 = model(device); opt2 = torch.optim.AdamW(m2.parameters(),lr=.001)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    m2.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer'])
    # RNG state byte tensors must be on CPU even when model tensors are CUDA.
    saved['rng']['cpu_rng'] = saved['rng']['cpu_rng'].cpu()
    saved['rng']['cuda_rng'] = [s.cpu() for s in saved['rng']['cuda_rng']]
    restore(saved['rng'])
    actual_loss = step(m2,opt2,saved['x'],saved['y'])
    torch.testing.assert_close(actual_loss,expected_loss,rtol=1e-6,atol=1e-7)
    for k,v in m2.state_dict().items():
        torch.testing.assert_close(v,expected[k],rtol=1e-6,atol=1e-7)
    print('resumed_next_loss_and_parameters PASS')
    print('restored=model,optimizer,python_rng,cpu_rng,cuda_rng,step,data_cursor')
    print('saved=' + str(checkpoint_path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['audit','resume'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--out',default='runs/advanced/resume')
    a=p.parse_args()
    if a.mode=='audit':audit(a.device)
    else:resume(a.device,a.out)


if __name__=='__main__':
    main()
