import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time
import lab5_lib

from skimage import io, data


def gaussian_pass_11812418(input_image, sigma):
    input_image = lab5_lib.FFT_zero_padding(input_image)
    row, col = input_image.shape

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    # define gaussian
    gaussian_filter = lab5_lib.generate_gaussian(row, col, sigma)
    plt.imshow(gaussian_filter)
    plt.show()
    filtered_img = np.multiply(input_image, gaussian_filter)
    output_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    output_img = lab5_lib.transform_centering(output_img)
    plt.imshow(np.real(output_img))
    plt.show()


if __name__ == '__main__':
    gaussian_pass_11812418(io.imread("Q5_2.tif"), 0.01)
