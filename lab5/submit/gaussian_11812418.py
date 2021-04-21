import time
import numpy as np
from PIL import Image
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
    gaussian_filter_HP = lab5_lib.generate_gaussian(row, col, sigma)
    gaussian_filter_LP = np.ones([row, col]) - lab5_lib.generate_gaussian(row, col, sigma)
    # print(gaussian_filter)
    # plt.imshow(gaussian_filter)
    # plt.show()
    filtered_img_HP = np.multiply(input_image, gaussian_filter_HP)
    output_img_HP = np.fft.ifft2(np.fft.ifftshift(filtered_img_HP))
    output_img_HP = lab5_lib.transform_centering(output_img_HP)
    filtered_img_LP = np.multiply(input_image, gaussian_filter_LP)
    output_img_LP = np.fft.ifft2(np.fft.ifftshift(filtered_img_LP))
    output_img_LP = lab5_lib.transform_centering(output_img_LP)
    # print(np.min(np.abs(output_img_HP)))
    # output_img_HP = lab5_lib.normalize(output_img_HP)
    # plt.imshow(np.real(output_img_HP))
    # plt.show()

    return np.abs(output_img_HP), np.abs(output_img_LP), lab5_lib.normalize(
        gaussian_filter_LP) * 255, lab5_lib.normalize(gaussian_filter_HP) * 255


if __name__ == '__main__':
    start_time = time.time()
    for i in [30, 60, 160]:
        A_LP, A_HP, G_HP, G_LP = gaussian_pass_11812418("Q5_2.tif", i)
        op_image_LP = Image.fromarray(A_LP.astype(np.uint8))
        op_image_LP.save("output/Q5_2_LP" + str(i) + ".tif")
        op_image_HP = Image.fromarray(A_HP.astype(np.uint8))
        op_image_HP.save("output/Q5_2_HP" + str(i) + ".tif")
        op_image_GL = Image.fromarray(G_LP.astype(np.uint8))
        op_image_GL.save("output/Q5_2_LP" + str(i) + "_F.tif")
        op_image_GH = Image.fromarray(G_HP.astype(np.uint8))
        op_image_GH.save("output/Q5_2_HP" + str(i) + "_F.tif")
    print("--- %s seconds ---" % (time.time() - start_time))
    for i in [30, 60, 160]:
        A_LP, A_HP, G_HP, G_LP = gaussian_pass_11812418("Q5_2.tif", i)
        op_image_LP = Image.fromarray(A_LP.astype(np.uint8))
        op_image_LP.save("output/Q5_2_LP" + str(i) + ".png")
        op_image_HP = Image.fromarray(A_HP.astype(np.uint8))
        op_image_HP.save("output/Q5_2_HP" + str(i) + ".png")
        op_image_GL = Image.fromarray(G_LP.astype(np.uint8))
        op_image_GL.save("output/Q5_2_LP" + str(i) + "_F.png")
        op_image_GH = Image.fromarray(G_HP.astype(np.uint8))
        op_image_GH.save("output/Q5_2_HP" + str(i) + "_F.png")

# fff男男女女男男女女nnnnnnnnnnnnnnnnffffffffffffnnnnnnnnnnnnnnn