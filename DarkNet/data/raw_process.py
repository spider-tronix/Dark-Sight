import numpy as np


def pack_raw(raw, black_level=512, nbits=14):
    """
    Packs Bayer Img to 4 Channels. The spatial resolution is reduced
    by a factor of 2 on each dimension. The black level is adjusted
    :param raw: Input RAW image
    :param black_level: Level of brightness at the darkest (black) part of a visual image
    :param nbits: Size of image
    :return: Processed image
    """
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (
            np.power(2, nbits) - 1 - black_level)  # Subtracting the black level and normalizing the image

    im = np.expand_dims(im, axis=2)  # Bring-up for depth-wise concatenation
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # Reduction of spatial dimension using a step size of '2'
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out
