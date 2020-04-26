#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

MIN_NORM = 1e-15
EPS = 1e-3

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + MIN_NORM, 1 - MIN_NORM)
        ctx.save_for_backward(x)
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1 + MIN_NORM)
        ctx.save_for_backward(x)
        # Change to double precition
        # z = x.double()
        # return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(MIN_NORM).log_().to(x.dtype)
        return (x + torch.sqrt_(x.pow(2) - 1)).clamp_min_(MIN_NORM).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5

def tanh(x, clamp=15):
    return x.clamp(-15, 15).tanh()

def artanh(x):
    return Artanh.apply(x)

def arcosh(x):
    return Arcosh.apply(x)

def hyp_distance(x, y):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xTy = torch.sum(x * y, dim=-1, keepdim=True)
    num = x2 + y2 - 2 * xTy
    den = (1 - x2) * (1 - y2)
    return arcosh(1 + 2 * num / den.clamp_min(MIN_NORM))

def mobius_add(x, y):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)

def expmap0(u):
    u_norm = u.norm(dim=-1, p=2, keepdim=True)
    gamma_1 = tanh(u_norm) * u / u_norm.clamp_min(MIN_NORM)
    return gamma_1
