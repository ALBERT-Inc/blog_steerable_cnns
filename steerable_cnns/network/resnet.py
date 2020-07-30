import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal

from steerable_cnns import group
from steerable_cnns.group import d4
from steerable_cnns.network.aug import random_augmentation
from steerable_cnns.network.convolution import SteerableConvolution2D
from steerable_cnns.network.posneg import posneg, posneg_representation


class ResBlock(chainer.Chain):
    def __init__(self, main_capsule, mid_capsule, n_in_capsules,
                 n_out_capsules, stride=1):
        super().__init__()

        mid_channels = n_out_capsules * mid_capsule.dim
        mid_capsule2 = posneg_representation(mid_capsule)
        out_channels = n_out_capsules * main_capsule.dim

        with self.init_scope():
            self.conv1 = SteerableConvolution2D((main_capsule, n_in_capsules),
                                                (mid_capsule, n_out_capsules),
                                                ksize=3, stride=stride, pad=1,
                                                nobias=True,
                                                initialW=HeNormal())
            self.norm1 = L.BatchNormalization(mid_channels * 2, eps=1e-5)
            self.activation1 = F.relu
            self.conv2 = SteerableConvolution2D((mid_capsule2, n_out_capsules),
                                                (main_capsule, n_out_capsules),
                                                ksize=3, stride=1, pad=1,
                                                nobias=True,
                                                initialW=HeNormal())
            self.norm2 = L.BatchNormalization(out_channels, eps=1e-5)
            self.activation2 = F.relu

            if n_in_capsules == n_out_capsules and stride == 1:
                return

            self.conv3 = SteerableConvolution2D((main_capsule, n_in_capsules),
                                                (main_capsule, n_out_capsules),
                                                ksize=1, stride=stride, pad=0,
                                                nobias=True,
                                                initialW=HeNormal())

    def forward(self, x, sc=None):
        h = self.activation1(self.norm1(posneg(self.conv1(x))))
        h = self.conv2(h)
        if sc is None:
            sc = self.conv3(x)
        sc = h + sc
        h = self.activation2(self.norm2(sc))
        return h, sc


class ResidualLayers(chainer.Chain):
    def __init__(self, main_capsule, mid_capsule, n_out_capsules, n_blocks):
        super().__init__()

        n_in_capsules = n_out_capsules // 2
        self.n_blocks = n_blocks

        with self.init_scope():
            self.b1 = ResBlock(main_capsule, mid_capsule, n_in_capsules,
                               n_out_capsules, stride=2)
            for i in range(1, n_blocks):
                block = ResBlock(main_capsule, mid_capsule, n_out_capsules,
                                 n_out_capsules)
                setattr(self, f'b{i+1}', block)

    def forward(self, x):
        h, sc = self.b1(x)
        for i in range(1, self.n_blocks):
            block = getattr(self, f'b{i+1}')
            h, sc = block(h, sc)
        return h


class InputLayers(chainer.Chain):
    def __init__(self, main_capsule, mid_capsule, n_out_capsules, n_blocks):
        super().__init__()

        n_in_capsules = n_out_capsules // 2
        in_channels = n_in_capsules * main_capsule.dim
        self.n_blocks = n_blocks

        with self.init_scope():
            self.conv0 = SteerableConvolution2D((d4.a1, None),
                                                (main_capsule, n_in_capsules),
                                                ksize=3, stride=1, pad=1,
                                                nobias=True,
                                                initialW=HeNormal())
            self.norm0 = L.BatchNormalization(in_channels, eps=1e-5)
            self.activation0 = F.relu

            self.b1 = ResBlock(main_capsule, mid_capsule, n_in_capsules,
                               n_out_capsules, stride=1)
            for i in range(1, n_blocks):
                block = ResBlock(main_capsule, mid_capsule, n_out_capsules,
                                 n_out_capsules)
                setattr(self, f'b{i+1}', block)

    def forward(self, x):
        h = self.activation0(self.norm0(self.conv0(x)))
        h, sc = self.b1(h)
        for i in range(1, self.n_blocks):
            block = getattr(self, f'b{i+1}')
            h, sc = block(h, sc)
        return h


class ResNet(chainer.Chain):
    def __init__(self, n_classes, n_blocks=2, width=20, augmentation=False):
        super().__init__()

        main_capsule = group.directsum_representation(
            (d4.regular, d4.qm, d4.qmr2, d4.qmr3))
        mid_capsule = group.directsum_representation(
            (d4.a1, d4.a2, d4.b1, d4.b2, d4.e, d4.e))

        with self.init_scope():
            self.l0 = InputLayers(main_capsule, mid_capsule, width, n_blocks)
            self.l1 = ResidualLayers(main_capsule, mid_capsule, 2 * width,
                                     n_blocks)
            self.l2 = ResidualLayers(main_capsule, mid_capsule, 4 * width,
                                     n_blocks)
            self.fc = L.Linear(4 * width * main_capsule.dim, n_classes)

        self.augmentation = augmentation

    def forward(self, x):
        if self.augmentation and chainer.config.train:
            x = random_augmentation(x)
        h = self.l0(x)
        h = self.l1(h)
        h = self.l2(h)
        h = F.mean(h, axis=(2, 3))
        return self.fc(h)
