#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class ManifoldParameter(torch.nn.Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """

    def __new__(cls, manifold, data=None, requires_grad=True):
        instance = torch.nn.Parameter._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    def __repr__(self):
        return '{} parameter containing:\n'.format(self.manifold.name) + super(torch.nn.Parameter, self).__repr__()


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()

    def distance(self, p1, p2):
        """Distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def expmap(self, u, p):
        """Projects tangent vector u in Tp on the manifold."""
        raise NotImplementedError

    def ptransp(self, x, y, u):
        """Parallel transport."""
        raise NotImplementedError

    def add(self, x, y, dim=-1):
        """Adds two hyperbolic points x and y."""
        raise NotImplementedError

    def inner(self, p, u, v=None):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError
