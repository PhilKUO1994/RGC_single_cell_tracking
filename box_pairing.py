import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def normalized_least_squares(patch1, patch2):
    product = np.mean((((patch1 - patch1.mean())/patch1.std()) - ((patch2 - patch2.mean())/patch2.std()))*(((patch1 - patch1.mean())/patch1.std()) - ((patch2 - patch2.mean())/patch2.std())))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 1
    else:
        #product /= stds
        return product

def box_pairing(im1,im2,ks,step_interval1,step_interval2):

    """
    searching features

    :param im1: left side,reference image
    :param im2:  right side image
    :param ks: sizes of searching kernel
    :param step_interval1: moving step size on reference images
    :param step_interval2: searching step size on right side image
    :return: coordinates of the same feature on both images
    """

    im1 = im1
    im2 = im2
    correlation_coefficient_arr = []
    max_cc = 0
    min_ls=0
    second_max_cc = 0
    for i in range(0,len(im1)-ks,step_interval1):
        print(i)
        for j in range(0,len(im1[0])-ks,step_interval1):
            if np.sum(im1[i:i+ks,j:j+ks])/255>200:
                for m in range(0,len(im2) - ks, step_interval2):
                    for k in range(0,int(len(im2[0])/2), step_interval2):
                        if np.sum(im2[m:m + ks, k:k + ks])/255>200:
                            cc = correlation_coefficient(im1[i:i+ks,j:j+ks],im2[m:m + ks, k:k + ks])
                            ls = normalized_least_squares(im1[i:i+ks,j:j+ks],im2[m:m + ks, k:k + ks])
                            if cc>max_cc:
                                max_cc = cc
                                #min_ls = ls
                                correlation_coefficient_arr=([i+ks/2,j+ks/2,m+ks/2,k+ks/2,im1[i:i+ks,j:j+ks],im2[m:m+ks, k:k+ks],ls])

    return correlation_coefficient_arr[:6]

im1 = cv2.imread('D:\\Code\\OCT_MouseCell_matching\\Rota_stitching\\examples1\\Y65_SoNar_OD_F1_Bas1_001.tif')
im2 = cv2.imread('D:\\Code\\OCT_MouseCell_matching\\Rota_stitching\\examples1\\Y65_SoNar_OD_F1_Day0_000.tif')

feedback = box_pairing(im1, im2, 50, 40, 40)
plt.imshow(feedback[-1])
plt.show()
plt.imshow(feedback[-2])
plt.show()