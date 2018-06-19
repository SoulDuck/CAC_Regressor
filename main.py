import tensorflow as tf
import numpy as np
from model import define_inputs , build_graph , sess_start
from cnn import algorithm
from data import next_batch
import matplotlib.pyplot as plt
import argparse
from data import batch_selector
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size' ,'-bs' , type = int  )
parser.add_argument('--init_lr' ,'-il' , type = float , default= 0.0001)
args=parser.parse_args()

max_iter = 100000
batch_size = args.batch_size
lr = args.init_lr
# Train
train_imgs_path = './cac_numpy_data/train_imgs.npy'
train_labs_path='./cac_numpy_data/train_labs.npy'

# Test
test_imgs_path = './cac_numpy_data/test_imgs.npy'
test_labs_path='./cac_numpy_data/test_labs.npy'

# Validation
val_imgs_path = './cac_numpy_data/val_imgs.npy'
val_labs_path='./cac_numpy_data/val_labs.npy'

train_imgs , train_labs, val_imgs , val_labs , test_imgs , test_labs = \
    map(np.load , [train_imgs_path , train_labs_path , val_imgs_path , val_labs_path , test_imgs_path , test_labs_path])
test_imgs=test_imgs/255.
val_imgs=val_imgs/255.


x_, y_, lr_, is_training, global_step = define_inputs(shape=[None, 540, 540, 3], n_classes=1)

logits = build_graph(x_=x_, y_=y_, is_training=is_training, aug_flag=True, actmap_flag=True, model='vgg_11',
                     random_crop_resize=540, bn=True)
sess, saver , summary_writer=sess_start(logs_path='./logs/vgg_11')
pred, pred_cls, cost_op, cost_mean,train_op, correct_pred, accuracy_op = algorithm(logits, y_, lr_, 'sgd', 'mse')



for i in range(100000):



    batch_xs, batch_ys = batch_selector(train_imgs, train_labs, 5, 5, 5, 5)
    #print np.shape(batch_xs) , np.shape(batch_ys)
    batch_xs = batch_xs / 255.
    batch_ys = batch_ys.reshape([-1, 1])
    train_fetches = [train_op, accuracy_op, cost_mean , logits ]
    train_feedDict = {x_: batch_xs, y_: batch_ys, lr_: lr, is_training: True}
    _ , train_acc, train_loss , train_preds = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
    values=batch_ys - train_preds
    indices=np.where([values  < 5 ])[0]
    rev_indices=np.where([values  > 5 ])[0]
    accuracy=len(batch_ys[indices]) / float(len(batch_ys))
    print accuracy

