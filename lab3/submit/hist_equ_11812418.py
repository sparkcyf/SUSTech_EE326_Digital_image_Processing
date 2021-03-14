import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def hist_equ(img_flat,bins):
    bins_arr = np.arange(bins + 1)
    # generate histogram
    histogram_in_data, histogram_in_index = np.histogram(img_flat, bins=bins_arr)

    # history equ
    # histogram Normalized
    histogram_in_data_normalized = histogram_in_data / np.sum(histogram_in_data)
    # historygram_accumulation
    histogram_in_data_accumulated = np.round(np.add.accumulate(histogram_in_data_normalized) * bins)

    img_after_hist_equ = histogram_in_data_accumulated[img_flat]

    histogram_out_data, histogram_out_index = np.histogram(img_after_hist_equ, bins=bins_arr)
    return histogram_out_data, img_after_hist_equ

def hist_equ_11812418(input_image):
    # Insert code here
    img = Image.open(input_image)
    img_arr = np.asarray(img)
    img_flat = img_arr.flatten() #flatten image to 1D array
    bins = 256
    input_hist, histogram_in_index = np.histogram(img_flat, bins=np.arange(bins + 1))

    output_hist, output_image = hist_equ(img_flat, bins)
    output_image = np.reshape(output_image, img_arr.shape)
    return (output_image, output_hist, input_hist)


def plot_hist(hist,filename,title):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(hist)
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# Q3_1_1
start_time = time.time()
(out_img2,hist_out,hist_in) = hist_equ_11812418("Q3_1_1.tif")

op_image = Image.fromarray(out_img2.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("output/img/hist_equ/Q3_1_1_M.tif")

plot_hist(hist_in,"output/img/hist_equ/Q3_1_1_hist.png","Histogram before histogram equalization")
plot_hist(hist_out,"output/img/hist_equ/Q3_1_1_M_hist.png","Histogram after histogram equalization")

# Q3_1_2
start_time = time.time()
(out_img2,hist_out,hist_in) = hist_equ_11812418("Q3_1_2.tif")

op_image = Image.fromarray(out_img2.astype(np.uint8))
print("--- %s seconds ---" % (time.time() - start_time))
op_image.save("output/img/hist_equ/Q3_1_2_M.tif")

plot_hist(hist_in,"output/img/hist_equ/Q3_1_2_hist.png","Histogram before histogram equalization")
plot_hist(hist_out,"output/img/hist_equ/Q3_1_2_M_hist.png","Histogram after histogram equalization")


