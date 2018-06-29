#-*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import sys
import time
import os
import random
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import aug
import tensorflow as tf
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--img_dir' , type = str)
args=parser.parse_args()

def make_tfrecord(tfrecord_path, resize ,*args ):
    """
    img source 에는 두가지 형태로 존재합니다 . str type 의 path 와
    numpy 형태의 list 입니다.
    :param tfrecord_path: e.g) './tmp.tfrecord'
    :param img_sources: e.g)[./pic1.png , ./pic2.png] or list flatted_imgs
    img_sources could be string , or numpy
    :param labels: 3.g) [1,1,1,1,1,0,0,0,0]
    :return:
    """
    if os.path.exists(tfrecord_path):
        print tfrecord_path + 'is exists'
        return
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    flag=True
    n_total =0
    counts = []
    for i,arg in enumerate(args):
        print 'Label :{} , # : {} '.format(i , arg[0])
        n_total += arg[0]
        counts.append(0)

    while(flag):
        label=random.randint(0,len(args)-1)
        n_max = args[label][0]
        if counts[label] < n_max:
            imgs = args[label][1]
            n_imgs = len(args[label][1])
            ind = counts[label] % n_imgs
            np_img = imgs[ind]
            counts[label] += 1
        elif np.sum(np.asarray(counts)) ==  n_total:
            for i, count in enumerate(counts):
                print 'Label : {} , # : {} '.format(i, count )
            flag = False
        else:
            continue;

        height, width = np.shape(np_img)[:2]

        msg = '\r-Progress : {0}'.format(str(np.sum(np.asarray(counts))) + '/' + str(n_total))
        sys.stdout.write(msg)
        sys.stdout.flush()
        if not resize is None:
            np_img = np.asarray(Image.fromarray(np_img).resize(resize, Image.ANTIALIAS))
        raw_img = np_img.tostring()  # ** Image to String **
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'raw_image': _bytes_feature(raw_img),
            'label': _int64_feature(label),
            'filename': _bytes_feature(tf.compat.as_bytes(str(ind)))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def extract_paths_cacs(patient_info , img_dir):

    # img_dir - pat_code - exam_date
    # patient_info = [[pat_code ,exam_date , cac_scores] , [pat_code ,exam_date , cac_scores] ... ]
    ret_paths = []
    ret_cacs = []
    for elements in patient_info:
        pat_code, exam_date, cac_score = elements
        tmp_paths = match_path2image(pat_code, exam_date, img_dir, 'png')
        ret_paths.extend(tmp_paths)
        ret_cacs.extend([cac_score] * len(tmp_paths))
    print '# of paths {}: '.format(len(ret_paths))
    print '# of cacs {}: '.format(len(ret_cacs))
    return ret_paths , ret_cacs



def paths2numpy(paths , savepath , preprocess_list = None):
    imgs= []
    for i,path in enumerate(paths):
        sys.stdout.write('\r progress {} {} '.format(i , len(paths)))
        sys.stdout.flush()
        img=Image.open(path)
        img=np.asarray(img)
        if 'projection' is preprocess_list:
            h,w,ch = np.shape(img)
            img=aug.fundus_projection(img, h)/255.
            plt.imsave('tmp.png' , img)
            img=np.asarray(Image.open('tmp.png').convert('RGB'))
            os.remove('tmp.png')
            if i ==0 :
                plt.imsave('projection_sample.png', img)
        imgs.append(img)
    imgs=np.asarray(imgs)
    if not savepath is None:
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
    elif data_id == '0100-0000003-020':
        # 0 - 150 , 150 명
        # 10 - inf : 150 , 150 명
        lab_0, lab_1 = [], []

        f = open(csv_path, 'r')

        for line in f.readlines():

            pat_code, cac_score, exam_date = line.split(',')[:3]
            cac_score = float(cac_score)
            if cac_score < 30:  # label 0
                lab_0.append([pat_code, exam_date, cac_score])
            elif cac_score < 1000000:
                lab_1.append([pat_code, exam_date, cac_score])
        return lab_0, lab_1

    elif data_id == '0100-0000003-021':
        # 0 - 150 , 150 명
        # 10 - inf : 150 , 150 명
        lab_0, lab_1 = [], []

        f = open(csv_path, 'r')

        for line in f.readlines():

            pat_code, cac_score, exam_date = line.split(',')[:3]
            cac_score = float(cac_score)
            if cac_score < 50:  # label 0
                lab_0.append([pat_code, exam_date, cac_score])
            elif cac_score < 1000000:
                lab_1.append([pat_code, exam_date, cac_score])
        return lab_0, lab_1

    elif data_id == '0100-0000003-022':
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

    elif data_id == '0100-0000003-023':
        # 0 - 150 , 200 명
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

    else:
        raise NotImplementedError




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


def make_data(data_id , img_dir ='/home/mediwhale/fundus_harddisk/merged_reg_fundus_540'):
    # data ID : 0100-0000003	018
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
        train_tfrecord_path = './train_0_10_11_inf.tfrecord'
        test_tfrecord_path = './test_0_10_11_inf.tfrecord'
        val_tfrecord_path = './val_0_10_11_inf.tfrecord'

        lab_1_test_paths, lab_1_test_cacs = extract_paths_cacs(lab_1_test[:], img_dir)
        lab_0_test_paths , lab_0_test_cacs = extract_paths_cacs(lab_0_test[:], img_dir)

        lab_1_val_paths, lab_1_val_cacs = extract_paths_cacs(lab_1_val[:], img_dir)
        lab_0_val_paths , lab_0_val_cacs = extract_paths_cacs(lab_0_val[:], img_dir)
        """
        f = open('./val_labels.txt' ,'w')
        for i,path in enumerate(lab_1_val_paths):
            name=os.path.splitext(os.path.split(path)[-1])[0]
            img = np.asarray(Image.open(path))
            plt.imsave('0100-0000003-019_label_1/{}.png'.format(name) , img)
            f.write(str(1)+'\n')


        for i,path in enumerate(lab_0_val_paths):
            name=os.path.splitext(os.path.split(path)[-1])[0]
            img = np.asarray(Image.open(path))
            plt.imsave('0100-0000003-019_label_0/{}.png'.format(name) , img)
            f.write(str(0)+'\n')
        exit()
        """
        lab_1_train_paths, lab_1_train_cacs = extract_paths_cacs(lab_1_train[:], img_dir)
        lab_0_train_paths , lab_0_train_cacs = extract_paths_cacs(lab_0_train[:], img_dir)
        if not os.path.exists(train_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_train_paths, None)
            imgs_1 = paths2numpy(lab_1_train_paths, None)
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_0), imgs_1))
        if not os.path.exists(test_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_val_paths, None)
            imgs_1 = paths2numpy(lab_1_val_paths, None)
            make_tfrecord(test_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1) )
        if not os.path.exists(val_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_test_paths, None)
            imgs_1 = paths2numpy(lab_1_test_paths, None)
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1))

    elif data_id == '0100-0000003-020':
        lab_0, lab_1 =sort_cac('merged_cacs_info_with_path.csv' , data_id)
        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, 75, 75)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, 75, 75)
        train_tfrecord_path = './0100-0000003-020/train_0_30_31_inf.tfrecord'
        test_tfrecord_path = './0100-0000003-020/test_0_30_31_inf.tfrecord'
        val_tfrecord_path = './0100-0000003-020/val_0_30_31_inf.tfrecord'

        lab_1_train_paths, lab_1_train_cacs = extract_paths_cacs(lab_1_train[:], img_dir)
        lab_0_train_paths , lab_0_train_cacs = extract_paths_cacs(lab_0_train[:], img_dir)

        lab_1_val_paths, lab_1_val_cacs = extract_paths_cacs(lab_1_val[:], img_dir)
        lab_0_val_paths , lab_0_val_cacs = extract_paths_cacs(lab_0_val[:], img_dir)

        lab_1_test_paths, lab_1_test_cacs = extract_paths_cacs(lab_1_test[:], img_dir)
        lab_0_test_paths , lab_0_test_cacs = extract_paths_cacs(lab_0_test[:], img_dir)

        if not os.path.exists(train_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_train_paths, None)
            imgs_1 = paths2numpy(lab_1_train_paths, None)
            make_tfrecord(train_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_0), imgs_1))
        if not os.path.exists(test_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_val_paths, None)
            imgs_1 = paths2numpy(lab_1_val_paths, None)
            make_tfrecord(test_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1) )
        if not os.path.exists(val_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_test_paths, None)
            imgs_1 = paths2numpy(lab_1_test_paths, None)
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1))

    elif data_id == '0100-0000003-021':
        lab_0, lab_1 =sort_cac('merged_cacs_info_with_path.csv' , data_id)
        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, 75, 75)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, 75, 75)
        train_tfrecord_path = './0100-0000003-021/train_0_50_51_inf.tfrecord'
        test_tfrecord_path = './0100-0000003-021/test_0_50_51_inf.tfrecord'
        val_tfrecord_path = './0100-0000003-021/val_0_50_51_inf.tfrecord'
        lab_1_train_paths, lab_1_train_cacs = extract_paths_cacs(lab_1_train[:], img_dir)
        lab_0_train_paths , lab_0_train_cacs = extract_paths_cacs(lab_0_train[:], img_dir)

        lab_1_val_paths, lab_1_val_cacs = extract_paths_cacs(lab_1_val[:], img_dir)
        lab_0_val_paths , lab_0_val_cacs = extract_paths_cacs(lab_0_val[:], img_dir)

        lab_1_test_paths, lab_1_test_cacs = extract_paths_cacs(lab_1_test[:], img_dir)
        lab_0_test_paths , lab_0_test_cacs = extract_paths_cacs(lab_0_test[:], img_dir)

        if not os.path.exists(train_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_train_paths, None)
            imgs_1 = paths2numpy(lab_1_train_paths, None)
            make_tfrecord(train_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_0), imgs_1))
        if not os.path.exists(test_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_val_paths, None)
            imgs_1 = paths2numpy(lab_1_val_paths, None)
            make_tfrecord(test_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1) )
        if not os.path.exists(val_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_test_paths, None)
            imgs_1 = paths2numpy(lab_1_test_paths, None)
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1))


    elif data_id == '0100-0000003-022':
        lab_0, lab_1 =sort_cac('merged_cacs_info_with_path.csv' , data_id)
        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, 75, 75)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, 75, 75)


        train_tfrecord_path = './0100-0000003-022/train_0_10_11_inf.tfrecord'
        test_tfrecord_path = './0100-0000003-022/test_0_10_11_inf.tfrecord'
        val_tfrecord_path = './0100-0000003-022/val_0_10_11_inf.tfrecord'

        lab_1_train_paths, lab_1_train_cacs = extract_paths_cacs(lab_1_train[:], img_dir)
        lab_0_train_paths , lab_0_train_cacs = extract_paths_cacs(lab_0_train[:], img_dir)

        lab_1_val_paths, lab_1_val_cacs = extract_paths_cacs(lab_1_val[:], img_dir)
        lab_0_val_paths , lab_0_val_cacs = extract_paths_cacs(lab_0_val[:], img_dir)

        lab_1_test_paths, lab_1_test_cacs = extract_paths_cacs(lab_1_test[:], img_dir)
        lab_0_test_paths , lab_0_test_cacs = extract_paths_cacs(lab_0_test[:], img_dir)

        if not os.path.exists(train_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_train_paths, None , 'projection')
            imgs_1 = paths2numpy(lab_1_train_paths, None ,'projection')
            make_tfrecord(train_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_0), imgs_1))
        if not os.path.exists(test_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_val_paths, None ,'projection')
            imgs_1 = paths2numpy(lab_1_val_paths, None ,'projection')
            make_tfrecord(test_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1) )
        if not os.path.exists(val_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_test_paths, None ,'projection')
            imgs_1 = paths2numpy(lab_1_test_paths, None,'projection')
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1))


    elif data_id == '0100-0000003-023':
        lab_0, lab_1 =sort_cac('merged_cacs_info_with_path.csv' , data_id) # lab_0 10174 , lab_1 3850


        lab_0_train, lab_0_val, lab_0_test = divide_paths_TVT(lab_0, n_val = 100, n_test = 1017)
        lab_1_train, lab_1_val, lab_1_test = divide_paths_TVT(lab_1, n_val = 100, n_test =385)


        train_tfrecord_path = './0100-0000003-022/train_0_10_11_inf.tfrecord'
        test_tfrecord_path = './0100-0000003-022/test_0_10_11_inf.tfrecord'
        val_tfrecord_path = './0100-0000003-022/val_0_10_11_inf.tfrecord'

        lab_1_train_paths, lab_1_train_cacs = extract_paths_cacs(lab_1_train[:], img_dir)
        lab_0_train_paths , lab_0_train_cacs = extract_paths_cacs(lab_0_train[:], img_dir)

        lab_1_val_paths, lab_1_val_cacs = extract_paths_cacs(lab_1_val[:], img_dir)
        lab_0_val_paths , lab_0_val_cacs = extract_paths_cacs(lab_0_val[:], img_dir)

        lab_1_test_paths, lab_1_test_cacs = extract_paths_cacs(lab_1_test[:], img_dir)
        lab_0_test_paths , lab_0_test_cacs = extract_paths_cacs(lab_0_test[:], img_dir)

        if not os.path.exists(train_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_train_paths, None , 'projection')
            imgs_1 = paths2numpy(lab_1_train_paths, None ,'projection')
            make_tfrecord(train_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_0), imgs_1))
        if not os.path.exists(test_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_val_paths, None ,'projection')
            imgs_1 = paths2numpy(lab_1_val_paths, None ,'projection')
            make_tfrecord(test_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1) )
        if not os.path.exists(val_tfrecord_path):
            imgs_0 = paths2numpy(lab_0_test_paths, None ,'projection')
            imgs_1 = paths2numpy(lab_1_test_paths, None,'projection')
            make_tfrecord(val_tfrecord_path, None, (len(imgs_0), imgs_0) , (len(imgs_1), imgs_1))


if '__main__' == __name__:


    # '/Volumes/Seagate Backup Plus Drive/IMAC/0100-0000003-016/merged_reg_fundus_540'
    img_dir = args.img_dir
    make_data(data_id='0100-0000003-023' , img_dir = img_dir)


