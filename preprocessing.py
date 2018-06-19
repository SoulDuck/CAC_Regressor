#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time
import os
import random
import glob
import csv
def match_path2image(  patient_id , exam_date , image_dir , extension ) :
    src_dir = os.path.join(image_dir , patient_id , exam_date ,'*.'+extension)
    paths =glob.glob(src_dir)
    return paths

def sort_cac(csv_path):
    # 0 - 75 , 75 명
    # 1-9  : 75 , 75 명
    # 10 - 99 : 75 , 75 명
    # 100 - 400 :  75 , 75 명
    # 400 : inf  - 75 , 75 명
    lab_0 , lab_1 , lab_2 , lab_3 , lab_4 = [],[],[],[],[]

    f = open(csv_path , 'r')
    for line in f.readlines():
        pat_code, cac_score, exam_date = line.split(',')[:3]
        cac_score =float(cac_score)
        if cac_score  < 1: # label 0
            lab_0.append([pat_code , exam_date , cac_score])
        elif cac_score < 10:
            lab_1.append([pat_code, exam_date, cac_score])
        elif cac_score < 100:
            lab_2.append([pat_code, exam_date, cac_score])
        elif cac_score < 1000000: # inf label 5
            lab_3.append([pat_code, exam_date, cac_score])

    return lab_0 , lab_1 , lab_2 ,lab_3


def divide_paths_TVT(paths, n_val , n_test , prefix =None ):
    val_paths=paths[:n_val]
    test_paths = paths[n_val: n_val + n_test]
    train_paths=paths[ n_val + n_test : ]

    print '{} trian : {} , val : {} test : {} '.format(prefix , len(train_paths) , len(val_paths) , len(test_paths))

    return train_paths , val_paths , test_paths


def crop_margin(image , resize ):
    """
    주변의 검정색을 지워 버립니다.
    :param path:
    :return:
    """

    """
    file name =1002959_20130627_L.png
    """
    start_time = time.time()
    im = image
    np_img = np.asarray(im)
    mean_pix = np.mean(np_img)
    pix = im.load()
    height, width = im.size  # Get the width and hight of the image for iterating over
    # pix[1000,1000] #Get the RGBA Value of the a pixel of an image
    c_x, c_y = (int(height / 2), int(width / 2))

    for y in range(c_y):
        if sum(pix[c_x, y]) > mean_pix:
            left = (c_x, y)
            break;

    for x in range(c_x):
        if sum(pix[x, c_y]) > mean_pix:
            up = (x, c_y)
            break;

    crop_img = im.crop((up[0], left[1], left[0], up[1]))

    #plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    if not resize is None:
        crop_img.resize(resize , Image.ANTIALIAS)
    end_time = time.time()
    return crop_img

if '__main__' == __name__:
    #match_image2label('merged_cacs_info_with_path.csv' , './' )
    lab_0, lab_1, lab_2, lab_3=sort_cac('merged_cacs_info_with_path.csv')
    lab_0_train , lab_0_val , lab_0_test=divide_paths_TVT(lab_0 , 75 , 75)
    lab_1_train , lab_1_val , lab_1_test=divide_paths_TVT(lab_1 , 75 , 75)
    lab_2_train , lab_2_val , lab_2_test=divide_paths_TVT(lab_2 , 75 , 75)
    lab_3_train , lab_3_val , lab_3_test=divide_paths_TVT(lab_3 , 75 , 75)


    img_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_540'
    train_cacs=[]
    train_paths = []

    for train_elements in [lab_0_train ,lab_1_train,lab_2_train,lab_3_train]:
        for elements in train_elements:
            pat_code, exam_date, cac_score = elements
            tmp_paths =match_path2image(pat_code , exam_date , img_dir , 'png')
            train_paths.extend(tmp_paths)
            train_cacs.extend([cac_score]*len(tmp_paths))


