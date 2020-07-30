from chainer.backends import cuda


def random_augmentation(x):
    """Apply basic augmentation (flip and shifts) to 32x32px RGB images.

    Specifically, this function horizontally flips each image with probability
    0.5, and shifts (or translates) each image with probability 0.5. If
    an image is shifted, horizontal and vertical shift sizes are uniformly
    chosen from (-4, -3, -2, -1, 1, 2, 3, 4). The shifted image is 0-padded.

    Args:
        x (cupy.ndarray): Images.

    Returns:
        y (cupy.ndarray): Augmented images.
    """
    if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
        raise ValueError

    rand = cuda.cupy.random.rand(len(x), dtype=cuda.cupy.float32)
    y = cuda.cupy.empty_like(x)
    cuda.elementwise(
        'raw T x, raw float32 rand', 'raw T y',
        '''
        int u = i % 32;
        int v = i / 32 % 32;
        int c = (int)(i / 32 / 32) % 3;
        int b = (int)(i / 32 / 32) / 3;

        unsigned char r = (unsigned char)(rand[b] * 256);
        bool do_flip = r >> 7;
        bool do_shift = (r >> 6) & 1;
        int shift_w = (r >> 3) & 7;
        int shift_h = r & 7;

        if (do_flip) u = 31 - u;
        if (do_shift) u += shift_w - 4 + (shift_w >> 2);
        if (do_shift) v += shift_h - 4 + (shift_h >> 2);

        ptrdiff_t i_in = ((ptrdiff_t)(b * 3 + c) * 32 + v) * 32 + u;
        T px = 0;
        if (0 <= u && u < 32 && 0 <= v && v < 32) px = x[i_in];
        y[i] = px;
        ''',
        'stcnn_random_aug')(x, rand, y, size=x.size)
    return y
