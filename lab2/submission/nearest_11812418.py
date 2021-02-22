'''

nearest interpolation
11812418

My SID ends with 8, so I need to enlarge the image for 1.8x (461) and shrink the image for 0.8x (205)
'''

import numpy as np
from PIL import Image

def nearest_11812418(input_file, dim):
    image = Image.open(input_file)
    imarray = np.array(image)
    #find the dimenssion of input file
    input_height = imarray.shape[0]
    input_width = imarray.shape[1]

    #set the output
    #dim(0) is the height
    #dim(1) is the width
    #init an array
    output_arr = np.zeros(dim)

    #iterate in output array
    for i in range(dim[0]):
        for j in range(dim[1]):
            #i height (dim0)
            #j width (dim1)
            interpolation_h = round((i)*(input_height-1)/(dim[0]-1))
            interpolation_w = round((j)*(input_width-1)/(dim[1]-1))
            output_arr[i][j] = imarray[interpolation_h][interpolation_w]


    return output_arr



#Testbench

#enlarge
dim1 = (461, 461)
output_img = nearest_11812418('rice.tif', dim1)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("enlarged_nearest_11812418.tif")

#shrink
dim2 = (205, 205)
output_img = nearest_11812418('rice.tif', dim2)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("shrunk_nearest_11812418.tif")
