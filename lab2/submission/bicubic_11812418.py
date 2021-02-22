'''

bicubic interpolation
11812418

My SID ends with 8, so I need to enlarge the image for 1.8x (461) and shrink the image for 0.8x (205)
'''

from scipy import interpolate
import numpy as np
from PIL import Image

def bicubic_11812418(input_file, dim):
    image = Image.open(input_file)
    imarray = np.array(image)

    # find the dimenssion of input file
    input_height = imarray.shape[0]
    input_width = imarray.shape[1]

    # set the output
    # dim(0) is the height
    # dim(1) is the width
    # init an array
    output_arr = np.zeros(dim)

    delta_height = 1
    delta_width = 1

    # base: output

    # construct the bicubic function
    def interp2d_bicubic_scipy(array_data, relative_y, relative_x):
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]
        f = interpolate.interp2d(y, x, array_data, kind='cubic')
        interp_result = f(relative_y, relative_x)
        return interp_result

    # iterate in output array
    for i in range(dim[0]):
        for j in range(dim[1]):
            # i height (dim0)
            # j width (dim1)

            # start from 1
            # transform to input coordinate
            projected_height = (i) * (input_height - 1) / (dim[0] - 1)
            projected_width = (j) * (input_width - 1) / (dim[1] - 1)
            # print(str(projected_height) + ' ' + str(projected_width))

            # find the border
            interpolation_x_floor = int(np.floor(projected_width))
            interpolation_y_floor = int(np.floor(projected_height))

            array16 = []
            frame_position_x = interpolation_x_floor
            frame_position_y = interpolation_y_floor
            # special case
            # XY MIN
            if ((interpolation_x_floor - 1 < 0) or (interpolation_y_floor - 1 < 0) or (
                    interpolation_x_floor + 2 > input_width - 1) or (interpolation_y_floor + 2 > input_height - 1)):

                border_X_Min = max(0, interpolation_x_floor - 1)
                border_Y_Min = max(0, interpolation_y_floor - 1)
                border_X_Max = min(input_width - 1, interpolation_x_floor + 2)
                border_Y_Max = min(input_height - 1, interpolation_y_floor + 2)

                if ((interpolation_x_floor - border_X_Min) < (border_X_Max - interpolation_x_floor - 1)):
                    # left border
                    frame_position_x = border_X_Min + 1
                elif ((interpolation_x_floor - border_X_Min) > (border_X_Max - interpolation_x_floor - 1)):
                    # right border
                    frame_position_x = border_X_Max - 2
                if ((interpolation_y_floor - border_Y_Min) < (border_Y_Max - interpolation_y_floor - 1)):
                    # lower border
                    frame_position_y = border_Y_Min + 1
                elif ((interpolation_y_floor - border_Y_Min) > (border_Y_Max - interpolation_y_floor - 1)):
                    # upper border
                    frame_position_y = border_Y_Max - 2

            for k in range(4):
                array16 = np.concatenate(
                    (array16, imarray[frame_position_y - 1 + k][frame_position_x - 1:frame_position_x + 3]), axis=None)

            output_arr[i][j] = interp2d_bicubic_scipy(array16, (projected_height - frame_position_y + 1),(projected_width - frame_position_x + 1))

    return output_arr


#Testbench

#enlarge
dim1 = (461, 461)
output_img = bicubic_11812418('rice.tif', dim1)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("enlarged_bicubic_11812418.tif")

#shrink
dim2 = (205, 205)
output_img = bicubic_11812418('rice.tif', dim2)
im = Image.fromarray(output_img.astype(np.uint8))
im.save("shrunk_bicubic_11812418.tif")