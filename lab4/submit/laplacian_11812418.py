import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time


def laplacian_11812418(input_img):
    pic_num = 1
    # Reading of the image into numpy array:
    img = Image.open(input_img)

    img_arr = np.asarray(img)

    row, col = img_arr.shape
    # Kernel for negative Laplacian:
    kernel = np.ones((3, 3)) * (-1)
    # laplacian
    kernel[1, 1] = 8
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # print(kernel)

    kernel_flat = kernel.flatten()

    # rewrite the conv in numpy

    # pad input img
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    input_img_padding = np.pad(img_arr, 1, pad_with, padder=0)

    # print(input_img_padding)

    # def par func
    def single_conv(i, j):
        im_arr = input_img_padding[(i - 1):(i + 2), (j - 1):(j + 2)].flatten()
        return np.sum(im_arr * kernel_flat)

    par_output = np.array(Parallel(n_jobs=6)(delayed(single_conv)(i, j)
                                             for i in range(1, row + 1)
                                             for j in range(1, col + 1)
                                             ))

    Lap = np.reshape(par_output, img_arr.shape)

    op_img = img_arr + Lap
    # Set negative values to 0, values over 255 to 255:
    op_img = np.clip(op_img, 0, 255)
    return op_img

start_time = time.time()
A = laplacian_11812418("Q4_1.tif")
op_image = Image.fromarray(A.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("Q4_1_laplacian.tif")

start_time = time.time()
A = laplacian_11812418("Q4_2.tif")
op_image = Image.fromarray(A.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("Q4_2_laplacian.tif")