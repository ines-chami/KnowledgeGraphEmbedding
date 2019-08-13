#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


# Euclidean utils

def batch_dot(x1, x2):
    return torch.sum(x1 * x2, dim=-1, keepdim=True)


def householder_reflection(x, v):
    v = v / torch.norm(v, p=2, dim=-1, keepdim=True)
    vTx = batch_dot(v, x)
    return x - 2 * vTx * v


def householder_rotation(x, v1, v2):
    v1 = v1 / torch.norm(v1, p=2, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, p=2, dim=-1, keepdim=True)
    v1Tx = batch_dot(v1, x)
    v2Tx = batch_dot(v2, x)
    v1Tv2 = batch_dot(v1, v2)
    return x - 2 * v1Tx * v1 - 2 * v2Tx * v2 + 4 * v1Tv2 * v2Tx * v1


# Hyperbolic utils

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)
