import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time

import lab6.lab6_lib
import lab6_lib as lab6
import median_filter_numba
import legacy_remove_noise


#Task3-1


img = Image.open("input/Q6_3_1.tiff")
input_image = np.asarray(img)
for j in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-12,1e-15,1e-20]:
    i = 0
    output_img = lab6.restore_planar_motion(input_image,0.1,0.1,1,j,0,i)
    op_image = Image.fromarray(output_img.astype(np.uint8))
    op_image.save("check/task3/Q6_3_1_wiener_"+str(i)+"_"+str(j)+".tiff")
    op_image.save("check/task3/Q6_3_1_wiener_"+str(i)+"_"+str(j)+".png")

img = Image.open("input/Q6_3_2_RT.png").convert('L')
# img = Image.open("input/Q6_3_3_RT.tiff").convert('L')
input_image = np.asarray(img)
for j in range(20):
    for i in [10,50,100]:
        output_img = lab6.restore_planar_motion(input_image,0.1,0.1,1,2.5*(10**-j),1,i)
        output_img = median_filter_numba.AdaptiveMedianFilter(output_img)
        output_img = median_filter_numba.AdaptiveMedianFilter(output_img)
        output_img = median_filter_numba.AdaptiveMedianFilter(output_img)
        op_image = Image.fromarray(output_img.astype(np.uint8))
        op_image.save("check/task3/3/Q6_3_3_wiener_"+str(i)+"_"+str(j)+".tiff")
        op_image.save("check/task3/3/Q6_3_3_wiener_"+str(i)+"_"+str(j)+".png")