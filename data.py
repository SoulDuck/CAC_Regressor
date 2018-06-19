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
