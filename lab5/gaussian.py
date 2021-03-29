import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time
import lab5_lib


def gaussian_pass_11812418(input_img, sigma):
    # Reading of the image into numpy array:
    img = Image.open(input_img)
    # FFT transform for img arr
    input_image_origin = np.asarray(img)
    input_image = np.asarray(img)

    input_image = lab5_lib.FFT_zero_padding(input_image)
    row, col = input_image.shape

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    # define gaussian
    gaussian_filter = lab5_lib.generate_gaussian(row, col, sigma)
    # print(gaussian_filter)
    # plt.imshow(gaussian_filter)
    # plt.show()
    filtered_img = np.multiply(input_image, gaussian_filter)
    output_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    output_img = lab5_lib.transform_centering(output_img)
    # print(np.min(np.abs(output_img)))
    # output_img = lab5_lib.normalize(output_img)
    # plt.imshow(np.real(output_img))
    # plt.show()

    return np.abs(output_img), np.abs(input_image_origin-output_img)


if __name__ == '__main__':
    start_time = time.time()
    A_LP,A_HP = gaussian_pass_11812418("Q5_2.tif", 30)
    op_image_LP = Image.fromarray(A_LP.astype(np.uint8))
    op_image_LP.save("output/Q5_2_LP30.tif")
    op_image_HP = Image.fromarray(A_HP.astype(np.uint8))
    op_image_HP.save("output/Q5_2_HP30.tif")
    A_LP,A_HP = gaussian_pass_11812418("Q5_2.tif", 60)
    op_image_LP = Image.fromarray(A_LP.astype(np.uint8))
    op_image_LP.save("output/Q5_2_LP60.tif")
    op_image_HP = Image.fromarray(A_HP.astype(np.uint8))
    op_image_HP.save("output/Q5_2_HP60.tif")
    A_LP,A_HP = gaussian_pass_11812418("Q5_2.tif", 160)
    op_image_LP = Image.fromarray(A_LP.astype(np.uint8))
    op_image_LP.save("output/Q5_2_LP160.tif")
    op_image_HP = Image.fromarray(A_HP.astype(np.uint8))
    op_image_HP.save("output/Q5_2_HP160.tif")


    print("--- %s seconds ---" % (time.time() - start_time))

