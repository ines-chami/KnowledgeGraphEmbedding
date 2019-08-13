#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .base import Manifold


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def init_weights(self, w, irange=1e-5):
        # TODO: xavier init
        w.data.uniform_(-irange, irange)
        return w

    def distance(self, p1, p2):
        return torch.sqrt((p1 - p2).pow(2).sum(dim=-1))

    def egrad2rgrad(self, p, dp):
        return dp

    def proj(self, p):
        return p

    def proj_tan(self, u, p):
        return u

    def expmap(self, u, p):
        return p + u

    def logmap(self, p1, p2):
        return p2 - p1

    def add(self, x, y):
        return x + y

    def inner(self, p, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v):
        return v
