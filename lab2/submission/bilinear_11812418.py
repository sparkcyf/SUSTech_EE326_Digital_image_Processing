'''

bilinear interpolation
11812418

My SID ends with 8, so I need to enlarge the image for 1.8x (461) and shrink the image for 0.8x (205)
'''

import numpy as np
from PIL import Image


def bilinear_11812418(input_file, dim):
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

    delta_height = 1
    delta_width = 1
    #base: output
    #iterate in output array
    for i in range(dim[0]):
        for j in range(dim[1]):
            #i height (dim0)
            #j width (dim1)

            #start from 1
            #transform to input coordinate
            projected_height = (i)*(input_height-1)/(dim[0]-1)
            projected_width = (j)*(input_width-1)/(dim[1]-1)
            #print(str(projected_height) + ' ' + str(projected_width))

            #find the border

            interpolation_h_up = int(np.ceil(projected_height))
            interpolation_h_down = int(np.floor(projected_height))
            interpolation_w_up = int(np.ceil(projected_width))
            interpolation_w_down = int(np.floor(projected_width))

            k_w_down = (abs(int(imarray[interpolation_h_up][interpolation_w_down])-int(imarray[interpolation_h_down][interpolation_w_down])))/delta_height

            w_down_val = int(imarray[interpolation_h_down][interpolation_w_down]) + k_w_down*(projected_height-interpolation_h_down)

            k_w_up = (abs(int(imarray[interpolation_h_up][interpolation_w_up])-int(imarray[interpolation_h_down][interpolation_w_up])))/delta_height
            w_up_val = int(imarray[interpolation_h_down][interpolation_w_up]) + k_w_up*(projected_height-interpolation_h_down)


            #special case
            if(projected_height % 1):
                w_down_val = imarray[int(projected_height)][int(projected_width)]
                w_up_val = imarray[int(projected_height)][int(projected_width)]

            if(projected_width % 1):
                interpolation_w_down = projected_width

            k_h = (abs(w_up_val-w_down_val)/delta_width)
            h_val = w_down_val + k_h*(projected_width-interpolation_w_down)

            output_arr[i][j] = h_val
    return output_arr

#Testbench

#enlarge
dim1 = (461, 461)
output_img = bilinear_11812418('rice.tif', dim1)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("enlarged_bilinear_11812418.tif")

#shrink
dim2 = (205, 205)
output_img = bilinear_11812418('rice.tif', dim2)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("shrunk_bilinear_11812418.tif")