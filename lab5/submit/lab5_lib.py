import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time


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
    x, y = np.meshgrid(np.linspace(0, a-1, a), np.linspace(0, b-1, b))
    x = x - a/2
    y = y - b/2
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


def normalize(input_img):
    return (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
