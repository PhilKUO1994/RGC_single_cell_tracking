"""
Author: Philip Guo yawenguo@connect.hku.hk
"""

import os
import cv2
import json
import numpy as np
from math import *
import math
import matplotlib.pyplot as plt
standard_length = 1000

def cal_angle(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    cos_value = np.cross(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    angle = acos(cos_value)/np.pi*180
    print("angle",angle)

def angle(v1, v2):
    # Vector 1 ; Vector 2
    v1 = [0,0,v1[0],v1[1]]
    v2 = [0,0,v2[0],v2[1]]
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = angle1-angle2
    else:
        included_angle = abs(angle1) + abs(angle2)
        #if included_angle > 180:
            #included_angle = 360 - included_angle
    return included_angle


def Put_on_Canvas(img):
    img = cv2.resize(img,(standard_length,standard_length))
    canvas = np.zeros((3*standard_length,3*standard_length,3),dtype=np.uint8)
    canvas[int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),:] =  img
    return canvas

def Move_img(img,x,y):
    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(img, T, (len(img[0]), len(img)))
    return img_translation


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def load_pos(json_path):
    # Reading data back
    with open(json_path, 'r') as f:
        data = json.load(f)
    #print(data["shapes"][0]["points"])
    return data["shapes"][0]["points"]

print("Contact Philip Guo (yawenguo@connect.hku.hk) if you have any questions！")

target_path = "./TargetImages/"
result_path = "./Output/"
black_path = "./Black/"
reference_path = "./Reference/"
lists = os.listdir(target_path)
img_paths = []
for file in lists:
    if ".tif" in file and file+'.json' in file:
        img_paths.append(file)

imgs_arr = []
pos_arr = []

reference_img = cv2.imread(target_path+img_paths[0])
raw_size = len(reference_img)
enlarge_scale = 1000/raw_size
reference_img = Put_on_Canvas(reference_img)
reference_pos = load_pos(target_path+img_paths[0][:-4]+".json")
reference_vector = np.array(reference_pos[1])-np.array(reference_pos[0])
reference_img = Move_img(reference_img,-reference_pos[0][0]*enlarge_scale+500,-reference_pos[0][1]*enlarge_scale+500)

final_img_arr = []
final_img_arr.append(reference_img)
for img_path in img_paths[1:]:
    if 'DS_Store' not in img_path:
        img = cv2.imread(target_path+img_path)
        raw_size = len(img)
        enlarge_scale = 1000 / raw_size
        img = Put_on_Canvas(img)
        pos = load_pos(target_path+img_path[:-4]+".json")
        vector = np.array(pos[1])-np.array(pos[0])
        rotate_angle = angle(reference_vector,vector) # 顺时针
        img = Move_img(img, -pos[0][0] * enlarge_scale + 500,
                                 -pos[0][1] * enlarge_scale + 500)
        img = rotate(img,-rotate_angle)
        final_img_arr.append(img)

for h in range(len(reference_img)):
    if np.sum(reference_img[h,:,:])>0:
        init_y = h
        break
for w in range(len(reference_img[0])):
    if np.sum(reference_img[:,w,:])>0:
        init_x = w
        break

#crop image : x = 125-375 375-625 625-875 y = 200-450  550-800
canvas = np.zeros((len(final_img_arr)*400,3200,3),dtype=np.uint8)
for i in range(len(final_img_arr)):
    reference_img = final_img_arr[i].copy()
    img1 = final_img_arr[i][init_y+200:init_y+500,init_x+100:init_x+400,:]
    reference_img = cv2.rectangle(reference_img, (init_x+100,init_y+200), (init_x+400,init_y+500), (255, 0, 0) , 2)
    print(img1.dtype)
    img2 = final_img_arr[i][init_y+200:init_y+500,init_x+350:init_x+650,:]
    reference_img = cv2.rectangle(reference_img, (init_x+350,init_y+200), (init_x+650,init_y+500), (255, 255, 0) , 2)
    img3 = final_img_arr[i][init_y+200:init_y+500,init_x+600:init_x+900,:]
    reference_img = cv2.rectangle(reference_img, (init_x+600,init_y+200), (init_x+900,init_y+500), (255, 0, 0) , 2)
    img4 = final_img_arr[i][init_y+500:init_y+800,init_x+100:init_x+400,:]
    reference_img = cv2.rectangle(reference_img, (init_x+100,init_y+500), (init_x+400,init_y+800), (0, 0, 255) , 2)
    img5 = final_img_arr[i][init_y+500:init_y+800,init_x+350:init_x+650,:]
    reference_img = cv2.rectangle(reference_img, (init_x+350,init_y+500), (init_x+650,init_y+800), (255, 0, 255) , 2)
    img6 = final_img_arr[i][init_y+500:init_y+800,init_x+600:init_x+900,:]
    reference_img = cv2.rectangle(reference_img, (init_x+600,init_y+500), (init_x+900,init_y+800), (0, 255, 0) , 2)
    img7 = final_img_arr[i][init_y+50:init_y+350,init_x+400:init_x+700,:]
    reference_img = cv2.rectangle(reference_img, (init_x+400,init_y+50), (init_x+700,init_y+350), (255, 0, 255) , 2)
    img8 = final_img_arr[i][init_y+650:init_y+950,init_x+400:init_x+700,:]
    reference_img = cv2.rectangle(reference_img, (init_x+400,init_y+650), (init_x+700,init_y+950), (255, 0, 0) , 2)
    cv2.imwrite(reference_path+img_paths[i]+".png",reference_img)
    imgs_arr = [img1,img2,img3,img4,img5,img6,img7,img8]
    for k in range(len(imgs_arr)):
        canvas[i*400:i*400+300,k*400:k*400+300,:] = imgs_arr[k]
        canvas = cv2.putText(canvas, "Period:"+str(i+1)+"  Pos:"+str(k+1), (k*400+20,i*400+350),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA) 
    for j in range(len(imgs_arr)):
        black_area = 0
        black_area = np.sum(imgs_arr[j][:,:,:]==[0,0,0])
        """
        CheckPoint1:
        """
        name1 = img_paths[i][:-4][:img_paths[i].index("_")]
        #name1 = "F2"
        #name2 = "_"+"F"+str(i+1)
        name2 = ""
        if black_area>=20:
            cv2.imwrite(black_path+ str(j+1)+"_" +name1+name2+img_paths[i][:-4][img_paths[i].index("_"):][:-4] + ".png",imgs_arr[j])
        else:
            cv2.imwrite(result_path+ str(j+1)+"_" +name1+name2+img_paths[i][:-4][img_paths[i].index("_"):]+".png",imgs_arr[j])
            
    '''
    cv2.imwrite(result_path+ "1_" +img_paths[i],img1)
    cv2.imwrite(result_path+ "2_"+img_paths[i],img2)
    cv2.imwrite(result_path+ "3_" +img_paths[i],img3)
    cv2.imwrite(result_path+ "4_" +img_paths[i],img4)
    cv2.imwrite(result_path+ "5_" +img_paths[i],img5)
    cv2.imwrite(result_path+ "6_" +img_paths[i],img6)
    cv2.imwrite(result_path+ "7_" +img_paths[i],img7)
    cv2.imwrite(result_path+ "8_" +img_paths[i],img8)
    '''
cv2.imwrite(reference_path+"Canvas"+".png",canvas)
print("==============Finished!")

