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

def paths2numpy(paths , savepath):
    imgs= []
    for path in paths:
        img=Image.open(path)
        img=np.asarray(img)
        imgs.append(img)
    imgs=np.asarray(imgs)
    np.save(file = savepath , arr = imgs)
    return imgs


def labs2numpy( labels , savepath):
    labels = np.asarray(labels)
    np.save(file=savepath, arr=labels)
    return labels

def match_path2image(  patient_id , exam_date , image_dir , extension ) :
    src_dir = os.path.join(image_dir , patient_id , exam_date ,'*.'+extension)
    paths =glob.glob(src_dir)
    return paths

def sort_cac(csv_path , data_id):
    if data_id == '0100-0000003-018':
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
    elif data_id == '0100-0000003-019':
        # 0 - 150 , 150 명
        # 10 - inf : 150 , 150 명
        lab_0, lab_1 = [], []

        f = open(csv_path, 'r')

        for line in f.readlines():

            pat_code, cac_score, exam_date = line.split(',')[:3]
            cac_score = float(cac_score)
            if cac_score < 10:  # label 0
                lab_0.append([pat_code, exam_date, cac_score])
            elif cac_score < 1000000:
                lab_1.append([pat_code, exam_date, cac_score])
        return lab_0, lab_1


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


def make_data(data_id):
    # data ID : 0100	0000003	018
    if data_id == '0100-0000003-018':
        #match_image2label('merged_cacs_info_with_path.csv' , './' )
        lab_0, lab_1, lab_2, lab_3=sort_cac('merged_cacs_info_with_path.csv')
        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, 75, 75)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, 75, 75)
        lab_2_train, lab_2_val, lab_2_test = divide_paths_TVT(lab_2, 75, 75)
        lab_3_train, lab_3_val, lab_3_test = divide_paths_TVT(lab_3, 75, 75)


        train_0_imgs=paths2numpy(lab_0_train ,savepath='./cac_numpy_data/train_0_lab.npy')
        test_0_imgs = paths2numpy(lab_0_test,savepath='./cac_numpy_data/test_0_lab.npy')
        val_0_imgs = paths2numpy(lab_0_val,savepath='./cac_numpy_data/val_0_lab.npy')

        train_1_imgs=paths2numpy(lab_1_train ,savepath='./cac_numpy_data/train_1_lab.npy')
        test_1_imgs = paths2numpy(lab_1_test,savepath='./cac_numpy_data/test_1_lab.npy')
        val_1_imgs = paths2numpy(lab_1_val,savepath='./cac_numpy_data/val_1_lab.npy')

        train_2_imgs=paths2numpy(lab_2_train ,savepath='./cac_numpy_data/train_2_lab.npy')
        test_2_imgs = paths2numpy(lab_2_test,savepath='./cac_numpy_data/test_2_lab.npy')
        val_2_imgs = paths2numpy(lab_2_val,savepath='./cac_numpy_data/val_2_lab.npy')

        train_3_imgs=paths2numpy(lab_3_train ,savepath='./cac_numpy_data/train_3_lab.npy')
        test_3_imgs = paths2numpy(lab_3_test,savepath='./cac_numpy_data/test_3_lab.npy')
        val_3_imgs = paths2numpy(lab_3_val,savepath='./cac_numpy_data/val_3_lab.npy')



        img_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_540'
        train_cacs = []
        train_paths = []
        for train_elements in [lab_0_train ,lab_1_train,lab_2_train,lab_3_train]:
            for elements in train_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths =match_path2image(pat_code , exam_date , img_dir , 'png')
                train_paths.extend(tmp_paths)
                train_cacs.extend([cac_score]*len(tmp_paths))
        print len(train_paths)
        print len(train_cacs)

        val_cacs = []
        val_paths = []
        for val_elements in [lab_0_val ,lab_1_val,lab_2_val,lab_3_val]:
            for elements in val_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths =match_path2image(pat_code , exam_date , img_dir , 'png')
                val_paths.extend(tmp_paths)
                val_cacs.extend([cac_score]*len(tmp_paths))
        print len(val_paths)
        print len(val_cacs)

        test_cacs = []
        test_paths = []
        for test_elements in [lab_0_test ,lab_1_test,lab_2_test,lab_3_test]:
            for elements in test_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths =match_path2image(pat_code , exam_date , img_dir , 'png')
                test_paths.extend(tmp_paths)
                test_cacs.extend([cac_score]*len(tmp_paths))

        if not os.path.exists('./train_imgs.npy'):
            train_imgs = paths2numpy(train_paths, './train_imgs.npy')
        if not os.path.exists('./test_imgs.npy'):
            test_imgs = paths2numpy(test_paths, './test_imgs.npy')
        if not os.path.exists('./val_imgs.npy'):
            val_imgs = paths2numpy(val_paths, './val_imgs.npy')


        if not os.path.exists('./train_labs.npy'):
            train_labs = labs2numpy(train_cacs, './train_labs.npy')
        if not os.path.exists('./test_labs.npy'):
            test_labs = labs2numpy(test_cacs, './test_labs.npy')
        if not os.path.exists('./val_labs.npy'):
            val_labs = labs2numpy(val_cacs, './val_labs.npy')
    elif data_id == '0100-0000003-019':
        lab_0, lab_1 =sort_cac('merged_cacs_info_with_path.csv' , data_id)

        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, 75, 75)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, 75, 75)


        img_dir = '/home/mediwhale/fundus_harddisk/merged_reg_fundus_540'
        train_cacs = []
        train_paths = []
        for train_elements in [lab_0_train, lab_1_train, lab_1_train,lab_1_train]:
            for elements in train_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths = match_path2image(pat_code, exam_date, img_dir, 'png')
                train_paths.extend(tmp_paths)
                train_cacs.extend([cac_score] * len(tmp_paths))
        print len(train_paths)
        print len(train_cacs)

        val_cacs = []
        val_paths = []
        for val_elements in [lab_0_val, lab_1_val]:
            for elements in val_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths = match_path2image(pat_code, exam_date, img_dir, 'png')
                val_paths.extend(tmp_paths)
                val_cacs.extend([cac_score] * len(tmp_paths))
        print len(val_paths)
        print len(val_cacs)

        test_cacs = []
        test_paths = []
        for test_elements in [lab_0_test, lab_1_test]:
            for elements in test_elements:
                pat_code, exam_date, cac_score = elements
                tmp_paths = match_path2image(pat_code, exam_date, img_dir, 'png')
                test_paths.extend(tmp_paths)
                test_cacs.extend([cac_score] * len(tmp_paths))
        print len(test_paths)
        print len(test_cacs)

        print test_paths


if '__main__' == __name__:
    make_data(data_id='0100-0000003-019')

