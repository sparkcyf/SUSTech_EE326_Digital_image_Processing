import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
import time


def hist_equ(img_flat, bins):
    bins_arr = np.arange(bins + 1)
    # generate histogram
    histogram_in_data, histogram_in_index = np.histogram(img_flat, bins=bins_arr)
    # history equ
    # histogram Normalized
    histogram_in_data_normalized = histogram_in_data / np.sum(histogram_in_data)
    # historygram_accumulation
    histogram_in_data_accumulated = np.round(np.add.accumulate(histogram_in_data_normalized) * bins)
    img_after_hist_equ = histogram_in_data_accumulated[img_flat -2]
    return img_after_hist_equ


def local_hist_equ_11812418(in_img, m_size):
    img = Image.open(in_img)
    in_img = np.asarray(img)
    row, col = in_img.shape
    out_img = np.zeros(in_img.shape, int)

    bin_num = 256
    bins_arr = range(bin_num + 1)

    histogram_in_data, histogram_in_index = np.histogram(in_img.flatten(), bins=bins_arr)

    # PAR
    def local_hist_equ_par(i, j):
        local_img = in_img[i:i + m_size, j:j + m_size]

        return (hist_equ(local_img.flatten(), bin_num)).reshape((m_size, m_size))

    local_hist_equ_par(1, 1)

    par_output = np.array(Parallel(n_jobs=1)(delayed(local_hist_equ_par)(i, j)
                                              for i in list(range(0, row - m_size, m_size)) + [row - m_size - 1]
                                              for j in list(range(0, col - m_size, m_size)) + [col - m_size - 1]
                                              ))

    # put array back

    for i in range(int(row / m_size)):
        for j in range(int(col / m_size)):
            out_img[i * m_size:(i * m_size + m_size), j * m_size:(j * m_size + m_size)] = par_output[
                int(i * row / m_size + j)]

    histogram_out_data, histogram_out_index = np.histogram(out_img.flatten(), bins=bins_arr)

    return (out_img, histogram_out_data, histogram_in_data)


def plot_hist(hist, filename, title):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(hist)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


start_time = time.time()
(out_img2, hist_out, hist_in) = local_hist_equ_11812418("Q3_3.tif", 16)

op_image = Image.fromarray(out_img2.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("output/img/local_hist_equal/Q3_3_M.tif")
op_image.save("output/img/local_hist_equal/Q3_3_M.png")

plot_hist(hist_in, "output/img/local_hist_equal/Q3_3_hist.png", "Histogram before local histogram equalization")
plot_hist(hist_out, "output/img/local_hist_equal/Q3_3_M_hist.png", "Histogram after local histogram equalization")
