import random
import numpy as np

def next_batch(batch_size , imgs, labs, fnames=None):
    indices = random.sample(range(np.shape(labs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    if not fnames is None:
        batch_fs = fnames[indices]
    else:
        batch_fs = None
    return batch_xs, batch_ys , batch_fs


def batch_selector(imgs ,labs  , batch_size_0 ,batch_size_1 ,batch_size_2 ,batch_size_3 ):
    ret_imgs = []
    # print batch_selector(val_labs)
    lab_0_indices = np.where((labs < 1))[0]
    #print len(lab_0_indices)

    lab_1_indices = np.where((labs < 9))[0]
    lab_1_indices = list((set(lab_1_indices) - set(lab_0_indices)))
    #print len(lab_1_indices)

    lab_2_indices = np.where((labs < 100))[0]
    lab_2_indices = list((set(lab_2_indices) - set(lab_0_indices) - set(lab_1_indices)))
    #print len(lab_2_indices)

    lab_3_indices = np.where((labs < 10000000))[0]
    lab_3_indices = list((set(lab_3_indices) - set(lab_2_indices) - set(lab_0_indices) - set(lab_1_indices)))
    #print len(lab_3_indices)
    assert len(labs) == len(lab_0_indices) + len(lab_1_indices) + len(lab_2_indices) + len(lab_3_indices)

    random.shuffle(lab_0_indices)
    random.shuffle(lab_1_indices)
    random.shuffle(lab_2_indices)
    random.shuffle(lab_3_indices)

    label_0_indices = lab_0_indices[:batch_size_0]
    label_1_indices = lab_1_indices[:batch_size_1]
    label_2_indices = lab_2_indices[:batch_size_2]
    label_3_indices = lab_3_indices[:batch_size_3]

    imgs = np.vstack([imgs[label_0_indices], imgs[label_1_indices], imgs[label_2_indices], imgs[label_3_indices]])
    labs = np.hstack([labs[label_0_indices], labs[label_1_indices], labs[label_2_indices], labs[label_3_indices]])
    return imgs , labs


if __name__ == '__main__':
    # Train
    train_imgs_path = './cac_numpy_data/train_imgs.npy'
    train_labs_path = './cac_numpy_data/train_labs.npy'

    # Test
    test_imgs_path = './cac_numpy_data/test_imgs.npy'
    test_labs_path = './cac_numpy_data/test_labs.npy'

    # Validation
    val_imgs_path = './cac_numpy_data/val_imgs.npy'
    val_labs_path = './cac_numpy_data/val_labs.npy'

    train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = \
        map(np.load, [train_imgs_path, train_labs_path, val_imgs_path, val_labs_path, test_imgs_path, test_labs_path])
    imgs , labs= batch_selector(train_imgs, train_labs, 10, 10, 10, 10)

