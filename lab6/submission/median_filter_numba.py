import numpy as np
from numba import njit, prange
from PIL import Image
import time


@njit
def padding(img, pad):
    padded_img = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    padded_img[pad:-pad, pad:-pad] = img
    return padded_img


@njit(parallel=True)
def AdaptiveMedianFilter(img, s=3, sMax=21):
    if len(img.shape) == 3:
        raise Exception("Single channel image only")

    row, col = img.shape
    a = sMax // 2
    padded_img = padding(img, a)

    f_img = np.zeros(padded_img.shape)

    for i in prange(a, row + a + 1):
        for j in prange(a, col + a + 1):
            value = stage_A(padded_img, i, j, s, sMax)
            f_img[i, j] = value

    return f_img[a:-a, a:-a]


@njit
def stage_A(mat, x, y, s, sMax):
    window = mat[x - (s // 2):x + (s // 2) + 1, y - (s // 2):y + (s // 2) + 1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return stage_B(window, Zmin, Zmed, Zmax)
    else:
        s += 2
        if s <= sMax:
            return stage_A(mat, x, y, s, sMax)
        else:
            return Zmed


@njit
def stage_B(window, Zmin, Zmed, Zmax):
    h, w = window.shape

    Zxy = window[h // 2, w // 2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0:
        return Zxy
    else:
        return Zmed


if __name__ == '__main__':
    img = Image.open("Q6_3_3_J.tiff")
    # FFT transform for img arr
    input_image_origin = np.asarray(img)
    start_time = time.time()
    output_img = AdaptiveMedianFilter(input_image_origin)
    print("--- %s seconds ---" % (time.time() - start_time))
    op_image_LP = Image.fromarray(output_img.astype(np.uint8))
    op_image_LP.save("output/Q6_3_3_J.tiff")
