import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time
import lab5_lib





def conv(input_img, kernel_flat):
    row, col = input_img.shape

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    input_img_padding = np.pad(input_img, 1, pad_with, padder=0)

    # def par func
    def single_conv(i, j):
        im_arr = input_img_padding[(i - 1):(i + 2), (j - 1):(j + 2)].flatten()
        return np.sum(im_arr * kernel_flat)

    par_output = np.array(Parallel(n_jobs=6)(delayed(single_conv)(i, j)
                                             for i in range(1, row + 1)
                                             for j in range(1, col + 1)
                                             ))

    Lap = np.reshape(par_output, input_img.shape)
    return Lap


def sobel_11812418(input_img):
    pic_num = 1
    # Reading of the image into numpy array:
    img = Image.open(input_img)

    # FFT transform for img arr
    img_arr = np.asarray(img)

    row, col = img_arr.shape

    # FFT Padding
    img_arr = lab5_lib.FFT_zero_padding(img_arr)

    img_arr = lab5_lib.multiply_center(img_arr)

    img_arr_fft = np.fft.fft2(img_arr)
    # plt.imshow(20*np.log(np.abs((np.fft.fft2(img_arr)))))
    # plt.show()

    kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    DFT_kernel_1_fft = lab5_lib.DFT_kernel(img_arr,kernel1)



    # K1
    filtered_1 = np.real(np.fft.ifft2(img_arr_fft * DFT_kernel_1_fft))
    filtered_1 = lab5_lib.FFT_extract_padding(filtered_1)
    filtered_1 = lab5_lib.multiply_center(filtered_1)
    # plt.imshow(filtered_1)
    # plt.show()



    # conv
    # op_img = np.fft.ifft2(conv(img_arr, kernel_flat))
    # print(filtered_1)
    op_img = lab5_lib.normalize(filtered_1)*255
    op_img = np.clip(op_img, 110, 140)
    print(op_img)

    plt.imshow(op_img)
    plt.show()

    return op_img


start_time = time.time()
A = sobel_11812418("Q5_1.tif")
op_image = Image.fromarray(A.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("output/Q5_1_M.tif")
