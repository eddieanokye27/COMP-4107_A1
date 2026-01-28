import math
import torch
import torch.nn as nn
from assignment1 import artificial_neuron, gradient_descent, pytorch_module

#Q1
def test_artificial_neuron():
    x = [1.0, -2.0, 3.0]
    w = [0.5, 1.0, -1.0]

    # manual computation
    y = 1.0*0.5 + (-2.0)*1.0 + 3.0*(-1.0)
    expected = y / (1 + math.exp(-y))

    out = artificial_neuron(x, w)

    print("Q1 output:", out)
    print("Q1 expected:", expected)
    assert abs(out - expected) < 1e-6

test_artificial_neuron()

#Q2
def f(v):
    return v[0]**2 + v[1]**2

def grad_f(v):
    return [2*v[0], 2*v[1]]

x0 = [5.0, -3.0]
alpha = 0.1

xmin, fmin = gradient_descent(f, grad_f, x0, alpha)

print("Q2 xmin:", xmin)
print("Q2 fmin:", fmin)

assert abs(xmin[0]) < 1e-3
assert abs(xmin[1]) < 1e-3
assert fmin < 1e-5

#Q3


def test_pytorch_module():
    model = pytorch_module()

    print("Q3 model:", model)
    assert isinstance(model, nn.Module)

    # check for Linear layer
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_linear, "Model must contain a Linear layer"

    # forward pass test
    x = torch.randn(1, 64)
    y = model(x)
    print("Q3 output shape:", y.shape)

test_pytorch_module()


