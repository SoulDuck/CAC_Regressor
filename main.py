import tensorflow as tf
import numpy as np
from model import define_inputs , build_graph , sess_start
from cnn import algorithm
from data import next_batch


# Train
train_imgs_path = './cac_numpy_data/train_imgs.npy'
train_labs_path='./cac_numpy_data/train_labs.npy'

# Test
test_imgs_path = './cac_numpy_data/test_imgs.npy'
test_labs_path='./cac_numpy_data/test_labs.npy'

# Validation
val_imgs_path = './cac_numpy_data/val_imgs.npy'
val_labs_path='./cac_numpy_data/val_labs.npy'

train_imgs , train_labs_, val_imgs , val_labs , test_imgs , test_labs = \
    map(np.load , [train_imgs_path , train_labs_path , val_imgs_path , val_labs_path , test_imgs_path , test_labs_path])
train_imgs=train_imgs/255.
test_imgs=test_imgs/255.
val_imgs=val_imgs/255.

x_, y_, lr_, is_training, global_step = define_inputs(shape=[None, 540, 540, 3], n_classes=1)

logits = build_graph(x_=x_, y_=y_, is_training=is_training, aug_flag=True, actmap_flag=True, model='vgg_11',
                     random_crop_resize=540, bn=True)
sess, saver , summary_writer=sess_start(logs_path='./logs/vgg_11')
pred, pred_cls, cost_op, train_op, correct_pred, accuracy_op = algorithm(logits, y_, lr_, 'sgd', 'mse')
max_iter = 100000
batch_size = 1
lr = 0.0001


batch_xs , batch_ys , batch_fs =next_batch(batch_size , train_imgs , train_labs_)
batch_ys=batch_ys.reshape([-1 ,1 ])
train_fetches = [train_op, accuracy_op, cost_op ]
train_feedDict = {x_: batch_xs, y_: batch_ys, lr_: lr, is_training: True}
_ , train_acc, train_loss = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
print train_loss
print train_acc


