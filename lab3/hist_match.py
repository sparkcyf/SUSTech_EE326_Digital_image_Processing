import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# define hist
hist = np.concatenate((np.linspace(0,10, 8 - 0),
                       np.linspace(10, 0.75, 16 - 8),
                       np.linspace(0.75, 0, 184 - 16),
                       np.linspace(0, 0.5, 200 - 184),
                       np.linspace(0.5, 0, 256 - 200)), axis=0)

spec_hist = np.zeros(256)




def hist_accumulation(hist, bins):
    histogram_in_data_normalized = hist / np.sum(hist)
    # historygram_accumulation
    histogram_in_data_accumulated = np.round(np.add.accumulate(histogram_in_data_normalized) * bins)

    return histogram_in_data_accumulated

def hist_normalization(hist, bins):
    histogram_in_data_normalized = hist / np.sum(hist)
    # historygram_accumulation

    return histogram_in_data_normalized


def hist_equ(img_flat, bins):
    bins_arr = np.arange(bins + 1)
    # generate histogram
    histogram_in_data, histogram_in_index = np.histogram(img_flat, bins=bins_arr)
    # history equ
    # histogram Normalized
    histogram_in_data_normalized = histogram_in_data / np.sum(histogram_in_data)
    # historygram_accumulation
    histogram_in_data_accumulated = np.round(np.add.accumulate(histogram_in_data_normalized) * bins)
    img_after_hist_equ = histogram_in_data_accumulated[img_flat]
    return histogram_in_data_accumulated, histogram_in_data, img_after_hist_equ


def hist_match(hist_in, hist_desired):
    hist_len = len(hist_in)
    hist_out = np.zeros(hist_len)

    def hist_match_index(i):
        index = min(range(hist_len), key=lambda j: abs(hist_desired[i] - hist_in[j]))
        hist_out[i] = index

    # def hist_match_index_par(i):
    #     return min(range(hist_len), key=lambda j: abs(hist_in[j] - hist_desired[i]))
    #
    # c = Parallel( n_jobs = 4 )( delayed( hist_match_index_par )( item ) for item in range(hist_len))
    # return c
    # init pool consume more time

    for i in range(hist_len):
        hist_match_index(i)
    return hist_out


def hist_match_11812418(input_image, spec_hist):
    # Insert code here
    img = Image.open(input_image)
    img_arr = np.asarray(img)
    img_flat = img_arr.flatten()  # flatten image to 1D array
    bins = 256

    # input img hist equ
    input_img_hist_after_hist_equ, input_hist, input_img_after_hist_equ = hist_equ(img_flat, bins)
    # desired_hist_normalized = hist_normalization(spec_hist, bins)
    desired_hist_accumulated = hist_accumulation(spec_hist, bins)
    # in: input_img_hist_after_hist_equ
    # desired: desired_hist_accumulated

    hist_lut = np.array(hist_match(input_img_hist_after_hist_equ, desired_hist_accumulated))

    plt.plot(input_img_hist_after_hist_equ)
    plt.show()
    plt.plot(desired_hist_accumulated)
    plt.show()

    img_after_hist_match = hist_lut[img_flat.astype(int) - 1]

    img_new_arr = np.reshape(input_img_after_hist_equ, img_arr.shape)
    output_hist, histogram_out_index = np.histogram(img_after_hist_match, bins=np.arange(bins + 1))

    return (img_new_arr, output_hist, input_hist)


def plot_hist(hist, filename, title):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(hist)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    # Q3_2
    start_time = time.time()
    (out_img2, hist_out, hist_in) = hist_match_11812418("Q3_2.tif", hist)

    op_image = Image.fromarray(out_img2.astype(np.uint8))
    print("--- %s seconds ---" % (time.time() - start_time))
    op_image.save("output/img/hist_matching/Q3_2_M.tif")
    op_image.save("output/img/hist_matching/Q3_2_M.png")

    plot_hist(hist_in, "output/img/hist_matching/Q3_2_hist.png", "Histogram before histogram matching")
    plot_hist(hist_out, "output/img/hist_matching/Q3_2_M_hist.png", "Histogram after histogram matching")
    plot_hist(hist, "output/img/hist_matching/hist.png", "Histogram used to match")

