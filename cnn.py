import tensorflow as tf

def convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k,k,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        layer=tf.nn.relu(layer , name='relu')
        if __debug__ == True:
            print 'layer name : ' ,name
            print 'layer shape : ' ,layer.get_shape()

        return layer

def max_pool(name,x , k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        if __debug__ ==True:

            layer=tf.nn.max_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)
            print 'layer name :', name
            print 'layer shape :', layer.get_shape()
    return layer
def avg_pool(name,x , k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        if __debug__ ==True:

            layer=tf.nn.avg_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)
            print 'layer name :', name
            print 'layer shape :', layer.get_shape()
    return layer

def affine(name,x,out_ch ,keep_prob):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc=tf.get_variable('w' , [height*width*in_ch ,out_ch] , initializer= tf.contrib.layers.xavier_initializer())
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc=tf.get_variable('w' ,[in_ch ,out_ch] ,initializer=tf.contrib.layers.xavier_initializer())

        b_fc=tf.Variable(tf.constant(0.1 ), out_ch)
        layer=tf.matmul(x , w_fc) + b_fc

        layer=tf.nn.relu(layer)
        layer=tf.nn.dropout(layer , keep_prob)
        print 'layer name :'
        print 'layer shape :',layer.get_shape()
        print 'layer dropout rate :',keep_prob
        return layer
def gap(name,x , n_classes ):
    in_ch=x.get_shape()[-1]
    gap_x=tf.reduce_mean(x, (1,2))
    with tf.variable_scope(name) as scope:
        gap_w = tf.get_variable('w', shape=[in_ch, n_classes], initializer=tf.random_normal_initializer(0, 0.01),
                                trainable=True)
    logits=tf.matmul(gap_x, gap_w , name='y_conv')
    return logits

def loss_selecter(loss_name , y_ ,y_conv):
    loss_name=loss_name.lower()

    if loss_name == 'ce': # Cross Entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_), name='cost')
    elif loss_name == 'll': # Meam Square Error
        cost = tf.reduce_mean(tf.losses.log_loss(predictions=y_conv, labels=y_), name='cost')
    elif loss_name == 'mse': # Log Loss
        cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=y_conv, labels=y_), name='cost')
    else:
        raise IndexError , 'Loss Must be '

    return cost

def optimizer_selecter(optimizer_name , learning_rate, cost_op  ):

    print 'optimizer option : GradientDescentOptimizer(default) | AdamOptimizer | momentum | '
    print 'selected optimizer : ', optimizer_name

    optimizer_dic = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer,
                     'momentum': tf.train.MomentumOptimizer}

    if not optimizer_name == 'momentum':
        train_op = optimizer_dic[optimizer_name](learning_rate).minimize(cost_op)
    else:
        train_op = tf.train.MomentumOptimizer(learning_rate , momentum = 0.9 , use_nesterov=True)
    return train_op


def algorithm(y_conv , y_ , learning_rate , optimizer_name='sgd' , loss_name = 'CE'):
    """
    :param y_conv: logits
    :param y_: labels
    :param learning_rate: learning rate
    :return:  pred,pred_cls , cost , correct_pred ,accuracy
    """
    if __debug__ ==True:
        print 'debug start : cnn.py | algorithm'
        print y_conv.get_shape()
        print y_.get_shape()


    pred = tf.nn.softmax(y_conv, name='softmax')
    pred_cls = tf.argmax(pred, axis=1, name='pred_cls')
    cost_op = loss_selecter(loss_name, y_, y_conv)
    train_op = optimizer_selecter(optimizer_name, learning_rate, cost_op)

    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')
    return pred, pred_cls, cost_op, train_op, correct_pred, accuracy


if __name__ == '__main__':
    print '__main__  : cnn.py'