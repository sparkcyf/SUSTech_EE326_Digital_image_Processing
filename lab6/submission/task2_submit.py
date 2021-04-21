import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import time

import lab6.lab6_lib
import lab6_lib as lab6
import median_filter_numba
import legacy_remove_noise


#Task2
start_time = time.time()
img = Image.open("input/Q6_2.tif")
input_image = np.asarray(img)
output_img = lab6.remove_air_turbulence(input_image,1,10e-7,1,100)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task2/Q6_2_R1.tiff")
op_image.save("check/task2/Q6_2_R1.png")


img = Image.open("input/Q6_2.tif")
input_image = np.asarray(img)
output_img = lab6.remove_air_turbulence(input_image,2,10e-7,1,100)
op_image = Image.fromarray(output_img.astype(np.uint8))
op_image.save("check/task2/Q6_2_R2.tiff")
op_image.save("check/task2/Q6_2_R2.png")


img = Image.open("input/Q6_2.tif")
input_image = np.asarray(img)
for i in [50,80,100]:
    for j in [1e-1,1e-4,1e-6]:
        output_img = lab6.remove_air_turbulence(input_image,3,j,1,i)
        op_image = Image.fromarray(output_img.astype(np.uint8))
        op_image.save("check/task2/Q6_2_wiener_"+str(i)+"_"+str(j)+".tiff")
        op_image.save("check/task2/Q6_2_wiener_"+str(i)+"_"+str(j)+".png")