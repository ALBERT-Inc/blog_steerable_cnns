import math

import chainer
import numpy as np
from chainer import functions
from chainer import initializers
from chainer import variable

from steerable_cnns import group


class SteerableConvolution2D(chainer.Link):
    def __init__(self, in_capsules, out_capsules, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None):
        super().__init__()

        pi, n_pi = in_capsules
        rho, n_rho = out_capsules
        pi_dim = pi.dim
        rho_dim = rho.dim

        pi = group.induced_representation(pi, ksize)
        basis = group.intertwiner_basis(pi, rho)
        basis = basis.astype(chainer.config.dtype)
        hom_dim = basis.shape[1]

        basis_randomizer = np.empty((hom_dim, hom_dim), dtype=basis.dtype)
        basis_scale = math.sqrt(basis.shape[0] / hom_dim)
        initializers.Orthogonal(scale=basis_scale)(basis_randomizer)
        basis = np.matmul(basis, basis_randomizer)

        basis = basis.reshape((rho_dim, pi_dim * ksize * ksize, hom_dim))
        self.add_persistent('basis', basis)

        if n_pi is None:
            in_channels = None
        else:
            in_channels = n_pi * pi_dim
        out_channels = n_rho * rho_dim

        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.in_channels = in_channels
        self.out_channels = out_channels

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)
            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        rho_dim, in_dim, hom_dim = self.basis.shape
        pi_dim = in_dim // (self.ksize * self.ksize)
        n_pi = in_channels // pi_dim
        n_rho = self.out_channels // rho_dim

        W_shape = (n_rho, n_pi, hom_dim)
        self.W.initialize(W_shape)
        # XXX: Assume initialization scheme is LeCun or MSRA init.
        # This tweak is incorrect if Glorot's init or orthogonal init used.
        self.W.array *= self.W.dtype.type(math.sqrt(hom_dim / in_dim))

        self.in_channels = in_channels

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        elif self.in_channels is None:
            self.in_channels = x.shape[1]
        W = functions.matmul(self.W[:, None], self.basis,
                             transa=False, transb=True)
        W = W.reshape((self.out_channels, self.in_channels,
                       self.ksize, self.ksize))
        return functions.convolution_2d(x, W, self.b, self.stride, self.pad)
