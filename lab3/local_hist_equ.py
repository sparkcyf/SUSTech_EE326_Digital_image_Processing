import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
def hist_equ(img_flat,bins):
    bins_arr = np.arange(bins + 1)
    # generate histogram
    histogram_in_data, histogram_in_index = np.histogram(img_flat, bins=bins_arr)
    # print(histogram_in_data)

    # history equ
    # histogram Normalized
    histogram_in_data_normalized = histogram_in_data / np.sum(histogram_in_data)
    # historygram_accumulation
    histogram_in_data_accumulated = np.round(np.add.accumulate(histogram_in_data_normalized) * bins)
    # plt.plot(histogram_in_data_accumulated)
    # plt.show()


    img_after_hist_equ = histogram_in_data_accumulated[img_flat]


    return histogram_in_data_accumulated, img_after_hist_equ


def local_hist_equ_11812418(input_image):
    # Insert code here
    img = Image.open(input_image)
    img_arr = np.asarray(img)
    img_flat = img_arr.flatten() #flatten image to 1D array
    bins = 256


    input_img_hist_after_hist_equ, input_img_after_hist_equ = hist_equ(img_flat, bins)
    plt.hist(input_img_after_hist_equ,256)
    plt.show()
    print(input_img_after_hist_equ)
    # #print for test
    # img_new_arr = np.reshape(input_img_after_hist_equ, img_arr.shape)
    # ouput_image = Image.fromarray(img_new_arr.astype(np.uint8))
    # ouput_image.save("output/0311-1.tif")



    return ()

def local_hist_equ(in_img, m_size):
    """
    Implement the local histogram equalization to the input images Q3_3.tif
    """
    img = Image.open(in_img)
    in_img = np.asarray(img)
    row, col = in_img.shape
    out_img = np.zeros(in_img.shape, int)

    bin_num = 256
    bins = range(bin_num + 1)
    for i in list(range(0, row - m_size, m_size)) + [row - m_size - 1]:
        # i = min(i, row - m_size)
        for j in list(range(0, col - m_size, m_size)) + [col - m_size - 1]:
            # j = min(j, col - m_size)
            local_img = in_img[i:i + m_size, j:j + m_size]
            local_hist, _ = np.histogram(local_img.flat, bins=bins, density=True)
            s = np.array([(bin_num - 1) * np.sum(local_hist[:k + 1]) for k in range(bin_num)])
            out_img[i:i + m_size, j:j + m_size] = np.array([s[r] for r in local_img], int).reshape((m_size, m_size))



    return out_img


# local_hist_equ_11812418("Q3_3.tif")
start_time = time.time()
out_img2 = local_hist_equ("Q3_3.tif",2)
ouput_image = Image.fromarray(out_img2.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
ouput_image.save("output/0311-2.tif")