import numpy as np


def contrast_stretch(img, q_lim_clip=0.025, scale_to_01=True):
    """
    Apply contrast stretch to an image.

    :param img: the image in numpy format
    :param q_lim_clip: the quantile to clip the values, symmetrically i.e. [q_lim_clip, 1 - q_lim_clip]
    :param scale_to_01: if True, scale the image to [0, 1] after clipping
    :return:
    """

    # add a new dimension for grayscale (will remove it at the end)
    if len(img.shape) == 2:
        img = img[:, :, None]

    for k in range(img.shape[-1]):
        # compute the limits
        _min = np.nanquantile(img[:, :, k], q_lim_clip)
        _max = np.nanquantile(img[:, :, k], 1 - q_lim_clip)

        if _min == _max:
            continue

        # clip
        img[:, :, k][img[:, :, k] < _min] = _min
        img[:, :, k][img[:, :, k] > _max] = _max

        # scale to [0, 1] if needed
        if scale_to_01:
            img[:, :, k] = (img[:, :, k] - _min) / (_max - _min)
            img[:, :, k][img[:, :, k] > 1] = 1.0  # rounding errors

    # remove the dummy dimension
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img
