import numpy as np
from PIL import Image
import time
import lab5_lib


def butterworse_pass_11812418(input_img, sigma, n):
    # Reading of the image into numpy array:
    img = Image.open(input_img)
    # FFT transform for img arr
    input_image = np.asarray(img)

    input_image = lab5_lib.FFT_zero_padding(input_image)
    row, col = input_image.shape

    input_image = np.fft.fft2(input_image)
    input_image = np.fft.fftshift(input_image)
    # 492,336

    # define butterworth
    butterworth_filter = np.ones([row, col])
    # = 1-lab5_lib.generate_butterworth(40, 40, 2, sigma)
    centers = [
        [109, 87],
        [109, 170],
        [115, 330],
        [115, 412],
        [227, 405],
        [227, 325],
        [223, 162],
        [223, 79]
    ]
    for point in centers:
        butterworth_filter = butterworth_filter - lab5_lib.generate_butterworth(row, col, n, sigma, point[0], point[1])

    input_image = np.multiply(input_image, butterworth_filter)

    # plt.imshow(np.log(np.abs(input_image)))
    # plt.show()

    # plt.imshow(input_image)
    # plt.show()
    filtered_img = input_image
    # filtered_img = np.multiply(input_image, butterworth_filter)
    output_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
    output_img = lab5_lib.transform_centering(output_img)
    output_img = lab5_lib.normalize(output_img) * 255
    # plt.imshow(np.real(output_img))
    # plt.show()

    return np.abs(output_img), np.log(np.abs(input_image))


if __name__ == '__main__':
    start_time = time.time()
    for sigma in [10, 40, 80, 120]:
        for n in [1, 2, 3, 4]:
            A, A_S = butterworse_pass_11812418("Q5_3.tif", sigma, n)
            op_image = Image.fromarray(A.astype(np.uint8))
            op_image.save("output/Q5_3_M_S" + str(sigma) + "_N" + str(n) + ".png")
            # plt.imshow(A_S)
            # plt.axis('off')
            # plt.savefig("output/Q5_3_M_F" + str(sigma) + "_N" + str(n) + ".png", bbox_inches='tight', pad_inches=0,
            #             dpi=150)
    print("--- %s seconds ---" % (time.time() - start_time))
