import math
import torch.nn as nn
from assignment1 import gradient_descent
#Q4a
#Define function and its gradient
x1, x2 = 3, -2
y1, y2 = 0.5, -0.75

def f_ab(v):
    a, b = v
    return 0.5 * (
        (a*x1 + b - y1)**2 +
        (a*x2 + b - y2)**2
    )

def grad_f_ab(v):
    a, b = v
    da = (a*x1 + b - y1)*x1 + (a*x2 + b - y2)*x2
    db = (a*x1 + b - y1) + (a*x2 + b - y2)
    return [da, db]

#run gradient descent
x0 = [0.0, 0.0]
alpha = 0.01

xmin, fmin = gradient_descent(f_ab, grad_f_ab, x0, alpha)

print("Q4a a, b:", xmin)
print("Q4a fmin:", fmin)

#Q4b
#Define SiLU activation and its derivative
def silu(x):
    return x / (1 + math.exp(-x))

def silu_prime(x):
    s = 1 / (1 + math.exp(-x))
    return s + x*s*(1 - s)

#Define function and its gradient
def f_ab_silu(v):
    a, b = v
    return 0.5 * (
        (silu(a*x1 + b) - y1)**2 +
        (silu(a*x2 + b) - y2)**2
    )

def grad_f_ab_silu(v):
    a, b = v

    z1 = a*x1 + b
    z2 = a*x2 + b

    da = (silu(z1) - y1) * silu_prime(z1) * x1 \
       + (silu(z2) - y2) * silu_prime(z2) * x2

    db = (silu(z1) - y1) * silu_prime(z1) \
       + (silu(z2) - y2) * silu_prime(z2)

    return [da, db]

#run gradient descent

x0 = [0.1, 0.1]
alpha = 0.01

xmin, fmin = gradient_descent(f_ab_silu, grad_f_ab_silu, x0, alpha)

print("Q4b a, b:", xmin)
print("Q4b fmin:", fmin)

