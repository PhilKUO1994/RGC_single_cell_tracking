import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

path_target_img = 'D:\\Code\\OCT_MouseCell_matching\\Rota_stitching\\examples1\\Y65_SoNar_OD_F1_Bas1_001.tif'
path_target_bashline = 'D:\\Code\\OCT_MouseCell_matching\\Rota_stitching\\examples1\\Y65_SoNar_OD_F1_Day0_000.tif'

standard_length = 1000



def Put_on_Canvas(img):
    img = cv2.resize(img,(standard_length,standard_length))
    canvas = np.zeros((3*standard_length,3*standard_length,3),dtype=np.uint8)
    canvas[int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),:] =  img
    return canvas

def transfer_img(path_target_img, path_target_bashline):

    # img2 is the baseline
    img1 = cv2.imread(path_target_img,0) # queryImage
    img1 = img1[250:1200,250:1200]
    img2 = cv2.imread(path_target_bashline,0) # trainImage
    img2 = img2[250:1200,250:1200]
    img3 = np.zeros((1200-250,1200-250))

    # Initiate SIFT detector
    sift = cv2.AKAZE_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)
    ref_matched_kpts = np.float32(np.array([kp1[m[0].queryIdx].pt for m in matches])).reshape(-1,1,2)
    sensed_matched_kpts = np.float32(np.array([kp2[m[0].trainIdx].pt for m in matches])).reshape(-1,1,2)

    # calculate homography
    #H = cv2.getPerspectiveTransform(ref_matched_kpts[:4],sensed_matched_kpts[:4])
    H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC,5.0)

    img1 = cv2.imread(path_target_img,0) # queryImage
    img2 = cv2.imread(path_target_bashline,0) # trainImage
    warped_image = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    return warped_image

if __name__ == '__main__':
    target_path = "./TargetImages/"
    result_path = "./Output/"
    black_path = "./Black/"
    reference_path = "./Reference/"
    regrouped_path = './GroupedImages/'

    lists = []
    counted_id = []
    for filename in os.listdir(target_path):
        if '.tif' in filename and '_Bas' in filename:
            tmp_name_list = []
            filename_id = filename[:filename.index('_Bas')]
            if filename_id not in counted_id:
                tmp_name_list.append(filename)
                for filename_searching in os.listdir(target_path):
                    if '.tif' in filename_searching and filename_searching != filename and filename_id in filename_searching:
                        tmp_name_list.append(filename_searching)
                lists.append(tmp_name_list)

    for list in lists:
        print(list)
        img_paths = []
        for file in list:
            if ".tif" in file:
                img_paths.append(file)
        final_img_arr = []
        final_name = []
        for img in list:
            if 'Bas' in img and '.tif' in img:
                reference_img = cv2.imread(target_path + img)
                reference_img = Put_on_Canvas(reference_img)
                reference_img_path = img
                final_img_arr.append(reference_img)
                final_name.append(img)
        for img in list:
            if 'Day' in img and '.tif' in img:
                print(img)
                compared_img = cv2.imread(target_path + img)
                compared_img = Put_on_Canvas(compared_img)
                compared_img = transfer_img(target_path + img, target_path + reference_img_path)
                compared_img = cv2.merge((compared_img, compared_img, compared_img))
                compared_img = Put_on_Canvas(compared_img)
                final_img_arr.append(compared_img)
                final_name.append(img)


        for h in range(len(reference_img)):
            if np.sum(reference_img[h, :, :]) > 0:
                init_y = h
                break
        for w in range(len(reference_img[0])):
            if np.sum(reference_img[:, w, :]) > 0:
                init_x = w
                break

        canvas = np.zeros((len(final_img_arr)*400,3200,3),dtype=np.uint8)
        for i in range(len(final_img_arr)):
            reference_img = final_img_arr[i].copy()
            img1 = final_img_arr[i][init_y+200:init_y+500,init_x+100:init_x+400,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+100,init_y+200), (init_x+400,init_y+500), (255, 0, 0) , 2)
            print(img1.dtype)
            img2 = final_img_arr[i][init_y+200:init_y+500,init_x+350:init_x+650,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+350,init_y+200), (init_x+650,init_y+500), (255, 255, 0) , 2)
            img3 = final_img_arr[i][init_y+200:init_y+500,init_x+600:init_x+900,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+600,init_y+200), (init_x+900,init_y+500), (255, 0, 0) , 2)
            img4 = final_img_arr[i][init_y+500:init_y+800,init_x+100:init_x+400,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+100,init_y+500), (init_x+400,init_y+800), (0, 0, 255) , 2)
            img5 = final_img_arr[i][init_y+500:init_y+800,init_x+350:init_x+650,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+350,init_y+500), (init_x+650,init_y+800), (255, 0, 255) , 2)
            img6 = final_img_arr[i][init_y+500:init_y+800,init_x+600:init_x+900,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+600,init_y+500), (init_x+900,init_y+800), (0, 255, 0) , 2)
            img7 = final_img_arr[i][init_y+50:init_y+350,init_x+400:init_x+700,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+400,init_y+50), (init_x+700,init_y+350), (255, 0, 255) , 2)
            img8 = final_img_arr[i][init_y+650:init_y+950,init_x+400:init_x+700,:]
            #reference_img = cv2.rectangle(reference_img, (init_x+400,init_y+650), (init_x+700,init_y+950), (255, 0, 0) , 2)
            path_name = reference_img_path[:reference_img_path.index('_Bas')]
            if not os.path.exists(reference_path + './' + path_name):
                os.mkdir(reference_path + './' + path_name)
            cv2.imwrite(reference_path + './' + path_name + '/' + final_name[i] +".png",reference_img)
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
                    path_name = reference_img_path[:reference_img_path.index('_Bas')]
                    if not os.path.exists(black_path + './' + path_name):
                        os.mkdir(black_path + './' + path_name)
                    cv2.imwrite(black_path + './' + path_name+ '/' +  str(j+1)+"_" +name1+name2+img_paths[i][:-4][img_paths[i].index("_"):]+".png",imgs_arr[j])
                else:
                    if not os.path.exists(result_path + './' + path_name):
                        os.mkdir(result_path + './' + path_name)
                    cv2.imwrite(result_path+ './' + path_name+ '/' + str(j+1)+"_" +name1+name2+img_paths[i][:-4][img_paths[i].index("_"):]+".png",imgs_arr[j])

        path_name = reference_img_path[:reference_img_path.index('_Bas')]
        if not os.path.exists(regrouped_path + '/' + path_name):
            os.mkdir(regrouped_path + '/' + path_name)
        for img_path in list:
            img = cv2.imread(target_path + img_path)
            cv2.imwrite('./'+ regrouped_path +'/' + path_name + '/' +img_path,img)
