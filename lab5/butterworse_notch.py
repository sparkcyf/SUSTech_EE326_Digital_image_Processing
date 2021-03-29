import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import lab5_lib


def butterworse_pass_11812418(input_img, sigma):
    # Reading of the image into numpy array:
    img = Image.open(input_img)
    # FFT transform for img arr
    input_image = np.asarray(img)

    input_image = lab5_lib.FFT_zero_padding(input_image)
    row, col = input_image.shape

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    #492,336

    # define butterworth
    butterworth_filter = 1-lab5_lib.generate_butterworth(40, 40, 2, sigma)
    bf_x = [70,150,310,395]
    bf_y = [90,210]

    for x in bf_x:
        for y in bf_y:
            input_image[x:x+40,y:y+40] = np.multiply(input_image[x:x+40,y:y+40],butterworth_filter)
    plt.imshow(np.log(np.abs(input_image)))
    plt.show()

    # plt.imshow(input_image)
    # plt.show()
    filtered_img = input_image
    # filtered_img = np.multiply(input_image, butterworth_filter)
    output_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    output_img = lab5_lib.transform_centering(output_img)
    output_img = lab5_lib.normalize(output_img)*255
    # plt.imshow(np.real(output_img))
    # plt.show()

    return np.abs(output_img)


if __name__ == '__main__':
    start_time = time.time()
    A = butterworse_pass_11812418("Q5_3.tif", 100)
    op_image = Image.fromarray(A.astype(np.uint8))
    print("--- %s seconds ---" % (time.time() - start_time))
    op_image.save("output/Q5_3_M.tif")
