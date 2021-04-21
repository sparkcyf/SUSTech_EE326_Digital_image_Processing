import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time
from numba import njit, prange


def FFT_zero_padding(input_img):
    row, col = input_img.shape
    output_image = np.zeros([2 * row, 2 * col])
    output_image[0:row, 0:col] = input_img
    return output_image


def FFT_extract_padding(input_img):
    row, col = input_img.shape
    output_image = input_img[int(row / 2):row, int(col / 2):col]
    return output_image


def multiply_center(input_img):
    row, col = input_img.shape
    I, J = np.ogrid[:row, :col]
    mask = np.full((row, col), -1)
    mask[(I + J) % 2 == 0] = 1
    return mask * input_img


def transform_centering(input_img):
    row, col = input_img.shape
    return input_img[0:int(row / 2), 0:int(col / 2)]


def DFT_kernel(input_image_padded, kernel):
    sz = (input_image_padded.shape[0] - kernel.shape[0], input_image_padded.shape[1] - kernel.shape[1])
    DFT_kernel_1 = np.pad(kernel, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    DFT_kernel_1 = multiply_center(DFT_kernel_1)
    DFT_kernel_1_fft = np.fft.fft2(DFT_kernel_1)
    return DFT_kernel_1_fft


def generate_gaussian(a, b, sigma):
    x, y = np.meshgrid(np.linspace(0, a - 1, a), np.linspace(0, b - 1, b))
    x = x - a / 2
    y = y - b / 2
    d = x * x + y * y
    g = np.exp(-(d / (2.0 * sigma ** 2)))
    # g = g/np.sum(g)
    return g


def generate_butterworth(row, col, n, sigma, cr, cc):
    x, y = np.meshgrid(np.linspace(0, col - 1, col), np.linspace(0, row - 1, row))
    x = x - cr
    y = y - cc
    d = np.sqrt(x * x + y * y)
    h = 1 / ((1 + (d / sigma)) ** (2 * n))
    return h


@njit
def normalize(input_img):
    return (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))


@njit
def padding_jit(img, pad):
    padded_img = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    padded_img[pad:-pad, pad:-pad] = img
    return padded_img


def normalize_pst(input_img):
    # find percentage
    p95 = np.percentile(input_img, 95)
    p5 = np.percentile(input_img, 5)
    input_img = np.clip(input_img, p5, p95)
    return (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def arithmetic_filter(i, j, image_after_padding, n_size):
    # im_arr = image_after_padding[(i - int((n_size-1)/2)):(i + int((n_size-1)/2)), (j - int((n_size-1)/2)):(j + int((n_size+1)/2))].flatten()
    im_arr = image_after_padding[(i):(i + int((n_size))), (j):(j + int((n_size)))].flatten()

    return np.sum(im_arr / (n_size ** 2))


@njit
def contrahamonic_filter(i, j, image_after_padding, n_size, Q):
    # im_arr = image_after_padding[(i - int((n_size-1)/2)):(i + int((n_size-1)/2)), (j - int((n_size-1)/2)):(j + int((n_size+1)/2))].flatten()
    im_arr = image_after_padding[(i):(i + int((n_size))), (j):(j + int((n_size)))].flatten()
    im_arr_Q = np.power(im_arr, Q)
    im_arr_QP1 = np.power(im_arr, Q + 1)
    out = 0
    # print(np.sum(im_arr_Q))
    if (np.sum(im_arr_QP1) != 0 and np.sum(im_arr_Q) != 0):
        out = np.sum(im_arr_QP1) / np.sum(im_arr_Q)
    # if (out == inf )
    return out


@njit
def alpha_trimmed_filter(i, j, image_after_padding, n_size, d):
    # im_arr = image_after_padding[(i - int((n_size-1)/2)):(i + int((n_size-1)/2)), (j - int((n_size-1)/2)):(j + int((n_size+1)/2))].flatten()
    im_arr = image_after_padding[(i):(i + int((n_size))), (j):(j + int((n_size)))].flatten()
    im_arr = im_arr[np.floor(d / 2):np.floor(n_size * n_size - d / 2)]
    return np.sum(im_arr / (n_size ** 2 - d))


def geometric_filter(i, j, image_after_padding, n_size):
    # im_arr = image_after_padding[(i - int((n_size-1)/2)):(i + int((n_size-1)/2)), (j - int((n_size-1)/2)):(j + int((n_size+1)/2))].flatten()
    im_arr = image_after_padding[(i):(i + int((n_size))), (j):(j + int((n_size)))].flatten()
    # print(im_arr)
    geo_product = np.prod(im_arr, where=im_arr > 0, dtype=np.float64)

    # zero_count = np.count_nonzero(im_arr)
    #
    # if(zero_count==0):
    #     return 0
    # # else:
    # print(str(np.power(geo_product,1/8))+"  "+str(np.mean(im_arr)))
    return np.power(geo_product, (1 / (n_size ** 2 - 1)))


# ContraharmonicMean
@njit(parallel=True)
def contrahamonic_mean_filter(input_img, n_size, Q):
    row, col = input_img.shape
    input_img_padding = padding_jit(input_img, int((n_size - 1) / 2))
    output_img = np.zeros((row, col))

    for i in prange(row):
        for j in prange(col):
            output_img[i][j] = contrahamonic_filter(i, j, input_img_padding, n_size, Q)
    output_img = np.where(np.isnan(output_img), 0, output_img)
    return output_img


@njit(parallel=True)
def alpha_trimmed_mean_filter(input_img, n_size, d):
    row, col = input_img.shape
    # print(input_img.shape)
    # input_img_padding = np.pad(input_img, n_size - 2, pad_with, padder=0)
    input_img_padding = padding_jit(input_img, int((n_size - 1) / 2))
    output_img = np.zeros((row, col))
    # print(input_img_padding.shape)

    for i in prange(row):
        for j in prange(col):
            output_img[i][j] = alpha_trimmed_filter(i, j, input_img_padding, n_size, d)

    # output_img = np.reshape(par_output, input_img.shape)
    return output_img


def arithmetic_mean_filter(input_img, n_size):
    row, col = input_img.shape
    input_img_padding = np.pad(input_img, n_size - 2, pad_with, padder=0)
    print(input_img_padding)

    # par_output = np.array(Parallel(n_jobs=6)(delayed(arithmetic_filter)(i, j,input_img_padding,n_size)
    #                                          for i in range(1, row + 1)
    #                                          for j in range(1, col + 1)
    #                                          ))
    output_img = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            output_img[i][j] = arithmetic_filter(i, j, input_img_padding, n_size)

    # output_img = np.reshape(par_output, input_img.shape)
    return output_img


def geometric_mean_filter(input_img, n_size):
    row, col = input_img.shape
    input_img_padding = np.pad(input_img, n_size - 2, pad_with, padder=0)
    # print(input_img_padding)

    # par_output = np.array(Parallel(n_jobs=6)(delayed(arithmetic_filter)(i, j,input_img_padding,n_size)
    #                                          for i in range(1, row + 1)
    #                                          for j in range(1, col + 1)
    #                                          ))
    output_img = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            output_img[i][j] = geometric_filter(i, j, input_img_padding, n_size)

    # output_img = np.reshape(par_output, input_img.shape)
    # print(normalize(output_img)*255)
    # output_img = normalize(output_img)
    return output_img


# REMOVE AIR TURBULENCE


def fft2d(input_img):
    # row, col = input_img.shape
    # input_img = FFT_zero_padding(input_img)
    # input_img = multiply_center(input_img)
    input_img = np.fft.fft2(input_img)
    input_img = np.fft.fftshift(input_img)

    return input_img


def ifft2d(input_img):
    output_img = np.fft.ifft2(np.fft.ifftshift(input_img))
    # output_img = transform_centering(output_img)
    return np.abs(output_img)


def ifft2d_real(input_img):
    output_img = np.fft.ifft2(np.fft.ifftshift(input_img))
    # output_img = transform_centering(output_img)
    return np.real(output_img)


def remove_air_turbulence(input_img, mode, K, gaussian, sigma):
    input_img_after_fft = fft2d(input_img)
    k = 0.0025

    row, col = input_img_after_fft.shape
    u, v = np.meshgrid(np.linspace(0, row - 1, row), np.linspace(0, col - 1, col))
    u = u - row / 2
    v = v - col / 2
    d = np.power(u, 2) + np.power(v, 2)
    H = np.exp(-(k * (np.power(d, 5 / 6))))

    if mode == 1:
        # full inverse filter
        output_img = input_img_after_fft / H

    if mode == 2:
        output_img = input_img_after_fft
        # Limit inverse
        for i in range(1, row + 1):
            for j in range(1, col + 1):
                if ((i - row / 2) ** 2 + (j - col / 2) ** 2) < 70:
                    output_img[i - 1, j - 1] = input_img_after_fft[i - 1, j - 1] / H[i - 1, j - 1]

    if mode == 3:
        # Wiener filter
        buf = np.power(H, 2)
        k2 = K
        output_img = input_img_after_fft * buf / (H * (buf + k2))
        if gaussian == 1:
            output_img = gaussian_LP_freq_filter(output_img, sigma)

    output_img = normalize(ifft2d(output_img)) * 255
    return output_img


def gaussian_LP_freq_filter(input_img, sigma):
    row, col = input_img.shape
    gaussian_filter_LP = generate_gaussian(row, col, sigma)
    filtered_img_LP = np.multiply(input_img, gaussian_filter_LP)
    return filtered_img_LP


def restore_planar_motion(input_img, a, b, T, k, LP, sigma):
    input_img_after_fft = fft2d(input_img)

    row, col = input_img.shape
    u, v = np.meshgrid(np.linspace(1, row, row), np.linspace(1, col, col))
    A = a * u + b * v

    # print(A.shape)
    T_arr = np.ones([row, col]) * T
    H = (T_arr / (np.pi * A)) * np.sin(A * np.pi) * np.exp(-1j * np.pi * A)
    # condition = np.array((A == 0))
    # H = H + condition
    buf = H * np.conj(H)
    # print(buf + k)
    F = input_img_after_fft * buf / (H * (buf + k))

    # gaussian
    if LP == 1:
        F = F * generate_gaussian(row, col, sigma)

    output_img = normalize_pst(ifft2d_real(F)) * 255

    return output_img
