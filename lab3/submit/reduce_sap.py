import time
import numpy as np
from PIL import Image
from joblib import Parallel, delayed


def reduce_SAP_11812418(input_image, n_size):
    img = Image.open(input_image)
    input_image = np.asarray(img)
    input_image_padding = np.copy(input_image)

    pad_num = int((n_size - 1) / 2)
    input_image_padding = np.pad(input_image_padding, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_padding.shape
    output_image = np.copy(input_image_padding)

    # filter_PAR
    def spacial_filter(i, j):
        a = np.median(input_image_padding[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])
        return a

    output_img_flat = Parallel(n_jobs=6)(delayed(spacial_filter)(i, j)
                                         for i in range(pad_num, m - pad_num)
                                         for j in range(pad_num, n - pad_num)
                                         )

    output_image = np.reshape(output_img_flat, input_image.shape)
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]

    return output_image


start_time = time.time()
img_new_arr = reduce_SAP_11812418("Q3_4.tif", 3)
op_image = Image.fromarray(img_new_arr.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("output/img/mid_filter/Q3_4_M.tif")
op_image.save("output/img/mid_filter/Q3_4_M.png")
