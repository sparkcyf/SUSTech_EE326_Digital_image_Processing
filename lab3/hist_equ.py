#https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def hist_arr(img_flat, bins):
    histogram = np.zeros(bins)
    for i in img_flat:
        histogram[i] += 1
    return histogram




def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)




def hist_equ_11812418(input_image):
    # Insert code here
    img = Image.open(input_image)
    img_arr = np.asarray(img)
    img_flat = img_arr.flatten() #flatten image to 1D array

    #hist
    # plt.hist(img_flat, bins=50)
    # plt.show()
    input_img_hist = hist_arr(img_flat, 256)
    #print(hist)

    #sum

    # cumulative sum
    # execute the fn
    cumulative_sum = cumsum(input_img_hist)

    # display the cs
    # plt.plot(cs)
    # plt.show()

    #normalize
    cs_norm = ((cumulative_sum - cumulative_sum.min()) * 255)/(cumulative_sum.max() - cumulative_sum.min())
    cs_norm = cs_norm.astype('uint8')


    # plt.plot(cs_norm)
    # plt.show()


    img_new = cs_norm[img_flat]
    #print(img_new)
    img_new_arr = np.reshape(img_new, img_arr.shape)
    output_img_hist = hist_arr(img_new_arr, 256)

    #image file
    ouput_image = Image.fromarray(img_new_arr.astype(np.uint8))
    #hist file



    # return (ouput_image,input_img_hist,output_img_hist)
    return (ouput_image,img_flat,img_new)










 #return (output_image, output_hist, input_hist)




(oimg,input_hist,output_hist) = hist_equ_11812418("Q3_1_2.tif")

oimg.save("output/Q3_1_1_M.tif")

plt.hist(input_hist, bins=256)
plt.savefig('output/input_hist.png')

plt.clf()
plt.hist(output_hist, bins=256)
plt.savefig('output/output_hist.png')


