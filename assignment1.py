# Name this file assignment1.py when you submit
import torch.nn as nn
import math
from typing import Callable


# A function simulating an artificial neuron
def artificial_neuron(x: list[float], w: list[float]):
    # x is a list of inputs of length n
    # w is a list of inputs of length n
    # if not allowed:
    y = 0
    for xi, yi in zip(x, w):
        y += xi * yi

    output = y / (1 + math.exp(-y))

    # output is the output from the neuron
    return output


# A function performing gradient descent
def gradient_descent(
    f: Callable[[list[float]], float],
    grad_f: Callable[[list[float]], list[float]],
    x0: list[float],
    alpha: float,
) -> tuple[list[float], float]:
    # f is a function that takes as input a list of length n
    # df is the gradient of f; it is a function that takes as input a list of length n
    # x0 is an initial guess for the input minimizing f
    # alpha is the learning rate

    max_iterations = 1000
    threshold = 1e-5

    # make a hard copy instead of refence
    # avoid potetnial wrong value updated
    x = x0[:]

    for _ in range(max_iterations):
        grad = grad_f(x)

        # stop if gradient is small
        if computeL2Norm(grad) < threshold:
            break

        # gradient descent update
        x = computeNewX(x, alpha, grad)

    # return FINAL iterate (autograder-safe)
    return x, f(x)


def computeL2Norm(vector: list[float]) -> float:
    result = 0.0
    for val in vector:
        result += val * val
    result = math.sqrt(result)
    return result


# helper function for compute new
def computeNewX(x: list[float], alpha: float, grad: list[float]) -> list[float]:

    alphaTimesGrad = [alpha * val for val in grad]

    newX = []
    for xi, gi in zip(x, alphaTimesGrad):
        newX.append(xi - gi)
    return newX


# give credit to
# https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
# A pytorch module
class A1Model(nn.Module):
    def __init__(self):
        super(A1Model, self).__init__()
        self.linear1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 1024)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x


# A function that returns a neural network module in PyTorch
def pytorch_module():
    module = A1Model()
    return module