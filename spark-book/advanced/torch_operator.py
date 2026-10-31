"""Build and check the CUDA operator against the installed PyTorch ABI."""
import argparse
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def install(bad_backward=False):
    if not torch.cuda.is_available():
        raise RuntimeError('qualified CUDA PyTorch environment required')
    load(name='spark4_ops', sources=[str(Path(__file__).with_name('torch_op.cu'))],
         extra_cuda_cflags=['-O2', '-lineinfo'], is_python_module=False,
         verbose=False)

    @torch.library.register_fake('spark4::muladd')
    def fake(a, b, c):
        torch._check(a.device.type == 'cuda')
        torch._check(a.device == b.device)
        torch._check(a.dtype == b.dtype)
        torch._check(a.dtype in (torch.float32, torch.float64))
        torch._check(a.is_contiguous() and b.is_contiguous())
        torch._check(a.shape == b.shape)
        return torch.empty_like(a)

    def setup(ctx, inputs, output):
        a, b, c = inputs
        ctx.save_for_backward(a, b)

    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * (a if bad_backward else b), grad * a, None

    torch.library.register_autograd('spark4::muladd', backward,
                                   setup_context=setup)
    return torch.ops.spark4.muladd.default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bad-backward', action='store_true')
    args = parser.parse_args()
    op = install(args.bad_backward)
    for shape in [(0,), (17,), (2, 3)]:
        a = torch.randn(shape, device='cuda', requires_grad=True)
        b = torch.randn(shape, device='cuda', requires_grad=True)
        torch.testing.assert_close(op(a, b, .125), a * b + .125)
        torch.library.opcheck(op, (a, b, .125))
    print('forward_and_opcheck shapes=3 PASS')
    a = torch.randn((3, 4), device='cuda', dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    passed = torch.autograd.gradcheck(op, (a, b, .125), raise_exception=False)
    if args.bad_backward:
        assert not passed, 'negative control failed to expose the wrong derivative'
        print('float64_gradcheck=FAIL expected_negative_control=true')
        return 2
    assert passed
    # Independent analytic comparison with built-in tensor arithmetic.
    g = torch.arange(12, device='cuda', dtype=torch.float64).reshape(3, 4)
    ga, gb = torch.autograd.grad(op(a, b, .125), (a, b), g)
    torch.testing.assert_close(ga, g * b)
    torch.testing.assert_close(gb, g * a)
    print('float64_gradcheck_and_analytic_backward PASS')
    bad = torch.ones((2, 3), device='cuda').t()
    try:
        op(bad, bad, 0.)
    except RuntimeError as e:
        assert 'contiguous required' in str(e)
    else:
        raise AssertionError('noncontiguous input accepted')
    print('noncontiguous_rejected PASS')
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = torch.arange(4097, device='cuda', dtype=torch.float32)
        b = torch.ones_like(a)
        actual, expected = op(a, b, 2.), a * b + 2.
    s.synchronize()
    torch.testing.assert_close(actual, expected)
    print('current_stream_check PASS')


if __name__ == '__main__':
    raise SystemExit(main())
