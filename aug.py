import tensorflow as tf
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa


def fundus_projection(img , scale):
    radius=scale/2
    blur_img=cv2.GaussianBlur(img,(0 ,0) , scale/30)
    merge_img=cv2.addWeighted(img , 4 , blur_img , -4 , 128)
    b = np.zeros(img.shape)
    cv2.circle(b , (radius,radius) , int(radius*0.9) , (1,1,1), -1 , 8 , 0 )
    merge_img = merge_img * b + 128 * (1 - b)
    return merge_img

def apply_projection(imgs , scale):
    imgs=map(lambda img : fundus_projection(img , scale) , imgs)
    return imgs

def clahe_equalized(img):
    if len(img.shape) == 2:
        img=np.reshape(img, list(np.shape(img)) +[1])
    assert (len(img.shape)==3)  #4D arrays
    img=img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.shape[-1] ==3: # if color shape
        for i in range(3):
            img[:, :, i]=clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
    elif img.shape[-1] ==1: # if Greys,
        img = clahe.apply(np.array(img, dtype = np.uint8))
    return img

def random_rotate_90(images):
    k=np.random.randint(0,4)
    images=np.rot90(images , k , axes =(1,2))
    return images

def random_rotate_with_PIL(image):

    ### usage: map(random_rotate , images) ###
    image=Image.fromarray(image)
    ind=random.randint(0,180)
    minus = random.randint(0,1)
    minus=bool(minus)
    if minus==True:
        ind=ind*-1
    img = image.rotate(ind)
    if __debug__ == True:
        print ind
    return np.asarray(img)

#==== histogram equalization
def histo_equalized(img):
    assert (len(np.shape(img))==2)  ,' image shape : {} '.format(np.shape(img)) #4D arrays
    return cv2.equalizeHist(np.array(img, dtype = np.uint8))

def aug_lv0(image_ , is_training , image_size , channel=3 , color =False):
    def aug_with_train(image, image_size):

        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, channel])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # Brightness / saturatio / constrast provides samll gains 2%~5% on cifar
        if color:
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.8)
        image = tf.image.per_image_standardization(image)
        return image

    def aug_with_test(image , image_size):

        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        return image

    image=tf.cond(is_training , lambda : aug_with_train(image_ , image_size=image_size)  , \
                  lambda  : aug_with_test(image_ , image_size=image_size) )
    return image

def aug_lv1(images):
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.ContrastNormalization((0.5, 1.5)),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        ]),
        iaa.Affine(scale=(0.8, 1.2), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-30, 30)),
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ]),
    ])
    augimgs = seq.augment_images(images)
    return augimgs



if __name__ == '__main__':
    img=Image.open('tmp/abnormal_actmap.png').convert('L')
    fig = plt.figure()
    ax=fig.add_subplot(132)
    HE_img = histo_equalized(img)
    ax.imshow(HE_img)
    ax = fig.add_subplot(133)
    rotated_img=random_rotate_with_PIL(img)
    ax.imshow(rotated_img)
    plt.show()