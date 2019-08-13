# !/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from manifolds import Manifold
from utils import artanh, tanh


class Poincare(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, c=1.0):
        super(Poincare, self).__init__()
        self.name = 'Poincare'
        self.min_norm = 1e-15
        self.c = torch.tensor([c])

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def distance(self, p1, p2):
        keepdim = False
        dim = -1
        sqrt_c = self.c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, dim=dim).norm(dim=dim, p=2, keepdim=keepdim)
        )
        dist = dist_c * 2 / sqrt_c
        return dist
        # return dist ** 2

    def _lambda_x(self, x):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - self.c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp):
        lambda_p = self._lambda_x(p)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x):
        c = torch.as_tensor(self.c).type_as(x)
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - 1e-3) / (c ** 0.5)
        if norm.is_cuda:
            maxnorm = maxnorm.to(norm.get_device())
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p):
        return u

    def expmap(self, u, p):
        c = torch.as_tensor(self.c).type_as(u)
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term)
        return gamma_1

    def add(self, x, y, dim=-1):
        c = torch.as_tensor(self.c).type_as(x)
        y = y + self.min_norm
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        # avoid division by zero in this way
        return num / denom.clamp_min(self.min_norm)

    def inner(self, x, u, v=None, keepdim=False, dim=-1):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x)
        return lambda_x ** 2 * (u * v).sum(dim=dim, keepdim=keepdim)

    def _gyration(self, u, v, w, dim: int = -1):
        c = c = torch.as_tensor(self.c).type_as(u)
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + self.c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def ptransp(self, x, y, u):
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y
