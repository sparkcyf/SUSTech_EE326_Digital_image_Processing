import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time

import lab6.lab6_lib
import lab6_lib as lab6
import median_filter_numba
import legacy_remove_noise

#Q611

#adaptive
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = median_filter_numba.AdaptiveMedianFilter(input_image)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_1_adaptive.tiff")
op_image.save("check/task1/Q6_1_1_adaptive.png")

#mean
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = legacy_remove_noise.reduce_SAP_11812418(input_image,3)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_1_mean.tiff")
op_image.save("check/task1/Q6_1_1_mean.png")

#contraharmonic_mean
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = lab6.contrahamonic_mean_filter(input_image,3,1.5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_1_contraharmonic.tiff")
op_image.save("check/task1/Q6_1_1_contraharmonic.png")

#Alpha trimmed mean
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = lab6.alpha_trimmed_mean_filter(input_image,5,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_1_alpha_trimmed.tiff")
op_image.save("check/task1/Q6_1_1_alpha_trimmed.png")

#Q612

#adaptive
img = Image.open("input/Q6_1_2.tiff")
input_image = np.asarray(img)
output_img = median_filter_numba.AdaptiveMedianFilter(input_image)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_2_adaptive.tiff")
op_image.save("check/task1/Q6_1_2_adaptive.png")

#mean
img = Image.open("input/Q6_1_2.tiff")
input_image = np.asarray(img)
output_img = legacy_remove_noise.reduce_SAP_11812418(input_image,3)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_2_mean.tiff")
op_image.save("check/task1/Q6_1_2_mean.png")

#contraharmonic_mean
img = Image.open("input/Q6_1_2.tiff")
input_image = np.asarray(img)
output_img = lab6.contrahamonic_mean_filter(input_image,3,-1.5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_2_contraharmonic.tiff")
op_image.save("check/task1/Q6_1_2_contraharmonic.png")

#Alpha trimmed mean
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = lab6.alpha_trimmed_mean_filter(input_image,5,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_2_alpha_trimmed.tiff")
op_image.save("check/task1/Q6_1_2_alpha_trimmed.png")

#Q613

#adaptive
img = Image.open("input/Q6_1_3.tiff")
input_image = np.asarray(img)
output_img = median_filter_numba.AdaptiveMedianFilter(input_image)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_3_adaptive.tiff")
op_image.save("check/task1/Q6_1_3_adaptive.png")

#mean
img = Image.open("input/Q6_1_3.tiff")
input_image = np.asarray(img)
output_img = legacy_remove_noise.reduce_SAP_11812418(input_image,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_3_mean.tiff")
op_image.save("check/task1/Q6_1_3_mean.png")

#contraharmonic_mean
img = Image.open("input/Q6_1_3.tiff")
input_image = np.asarray(img)
output_img = lab6.contrahamonic_mean_filter(input_image,3,1.5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_3_contraharmonic.tiff")
op_image.save("check/task1/Q6_1_3_contraharmonic.png")

#Alpha trimmed mean
img = Image.open("input/Q6_1_3.tiff")
input_image = np.asarray(img)
output_img = lab6.alpha_trimmed_mean_filter(input_image,5,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_3_alpha_trimmed.tiff")
op_image.save("check/task1/Q6_1_3_alpha_trimmed.png")

#Q614

#adaptive
img = Image.open("input/Q6_1_4.tiff")
input_image = np.asarray(img)
output_img = median_filter_numba.AdaptiveMedianFilter(input_image)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_4_adaptive.tiff")
op_image.save("check/task1/Q6_1_4_adaptive.png")

#mean
img = Image.open("input/Q6_1_4.tiff")
input_image = np.asarray(img)
output_img = legacy_remove_noise.reduce_SAP_11812418(input_image,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_4_mean.tiff")
op_image.save("check/task1/Q6_1_4_mean.png")

#contraharmonic_mean
img = Image.open("input/Q6_1_4.tiff")
input_image = np.asarray(img)
output_img = lab6.contrahamonic_mean_filter(input_image,3,1.5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_4_contraharmonic.tiff")
op_image.save("check/task1/Q6_1_4_contraharmonic.png")

#Alpha trimmed mean
img = Image.open("input/Q6_1_1.tiff")
input_image = np.asarray(img)
output_img = lab6.alpha_trimmed_mean_filter(input_image,5,5)
output_img = lab6.normalize(output_img)*255
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task1/Q6_1_4_alpha_trimmed.tiff")
op_image.save("check/task1/Q6_1_4_alpha_trimmed.png")

