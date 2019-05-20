#coding:utf-8

#pip install tflearn

from tflearn.datasets import oxflower17
import tensorflow as tf
import sys
# import vgg_11_lrn as vg


#加载数据集
X, Y = oxflower17.load_data(one_hot = True)
#查看数据的信息
print(X.shape)
print(Y.shape)

#构建与训练相关的参数
# trainEpoch
train_epoch = 10
batch_size = 16 #一般来说这个值是16的整数倍
display_epoch = 100 

#构建vgg网络所需要的参数及网络结构
n_class = Y.shape[1]
lr = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None , n_class])

def vgg_network():
    '''设计网络权重、偏置以及所使用这些变量进行的卷积、池化和全连接过程'''
    weight = {
        'wc1_1':tf.get_variable('wc1_1', [3, 3, 3, 64]),
        'wc2_1':tf.get_variable('wc2_1', [3, 3, 64, 128]),
        'wc3_1':tf.get_variable('wc3_1', [3, 3, 128, 256]),
        'wc3_2':tf.get_variable('wc3_2', [3, 3, 256, 256]),
        'wc4_1':tf.get_variable('wc4_1', [3, 3, 256, 512]),
        'wc4_2':tf.get_variable('wc4_2', [3, 3, 512, 512]),
        'wc5_1':tf.get_variable('wc5_1', [3, 3, 512, 512]),
        'wc5_2':tf.get_variable('wc5_2', [3, 3, 512, 512]),
        
        'wfc_1':tf.get_variable('wfc_1', [7*7*512, 4096]),
        'wfc_2':tf.get_variable('wfc_2', [4096, 4096]),
        'wfc_3':tf.get_variable('wfc_3', [4096, n_class])
        }
    
    biase = {
        'bc1_1':tf.get_variable('bc1_1', [64]),
        'bc2_1':tf.get_variable('bc2_1', [128]),
        'bc3_1':tf.get_variable('bc3_1', [256]),
        'bc3_2':tf.get_variable('bc3_2', [256]),
        'bc4_1':tf.get_variable('bc4_1', [512]),
        'bc4_2':tf.get_variable('bc4_2', [512]),
        'bc5_1':tf.get_variable('bc5_1', [512]),
        'bc5_2':tf.get_variable('bc5_2', [512]),
        'bfc_1':tf.get_variable('bfc_1', [4096]),
        'bfc_2':tf.get_variable('bfc_2', [4096]),
        'bfc_3':tf.get_variable('bfc_3', [n_class])
        }
    
    #############
    #conv_1
    #############
    #定义动态卷积、池化、全连接过程(参数如何使用)
    #卷积过程
    net = tf.nn.conv2d(input=x, filter=weight['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc1_1']))
    #局部响应归一化
    net = tf.nn.lrn(net)
    #池化
    net = tf.nn.max_pool(value=net , ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'SAME')
    
    #############
    #conv_2
    #############
    #卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc2_1']))
    #池化
    net = tf.nn.max_pool(value=net , ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'VALID')

    #############
    #conv_3
    #############
    #卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_1']))
    net = tf.nn.conv2d(input=net, filter=weight['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_2']))
    #池化
    net = tf.nn.max_pool(value=net , ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'VALID')

    #############
    #conv_4
    #############
    #卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_1']))
    net = tf.nn.conv2d(input=net, filter=weight['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_2']))
    #池化
    net = tf.nn.max_pool(value=net , ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'VALID')
    
    #############
    #conv_5
    #############
    #卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_1']))
    net = tf.nn.conv2d(input=net, filter=weight['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_2']))
    #池化
    net = tf.nn.max_pool(value=net , ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding = 'VALID')
    
    #构建全连接部分的网络结构
    #拉升
    net = tf.reshape(net, shape = [-1, weight['wfc_1'].get_shape()[0]])
    
    ######################
    #fc1
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_1']) + biase['bfc_1'])
    
    ######################
    #fc2
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_2']) + biase['bfc_2'])
    
    ######################
    #fc3(out)
    ######################
    #注意：全连接的最后一层输出所使用的激活函数是likelyhood-softmax
    return tf.matmul(net, weight['wfc_3']) +biase['bfc_3']

#设计损失函数及优化器， 并且建立评估函数
######################
#Loss Function & Optimizer
######################
pred = vgg_network()
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = pred)
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
######################
#evaluation
######################
acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32), axis = None)

######################
#train—beginning
######################
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    learning_rate = 0.01
    for epoch in range(train_epoch):
        
        total_batch = X.shape[0]//batch_size #104
        for i in range(total_batch):
            X_train = X[i*batch_size : i*batch_size + batch_size]
            Y_train = Y[i*batch_size : i*batch_size + batch_size]
            print('step:%s'%str(epoch)+' '+str(i))
            sess.run(opt, feed_dict = {x:X_train, y:Y_train, lr:learning_rate})  
            
        ######################
        #evaluation
        ######################
        if (epoch + 1) % display_epoch == 0:
            cost, accuaray = sess.run([loss, acc])
        
            print('step:%s, loss:%s, acc:%s'%(str(epoch+1) + '-' + str(i),\
                                             cost[0], accuaray))
            ######################
            #learning_rate change
            ######################
            learning_rate = 0.01*(1-epoch/train_epoch)**2
        save = tf.train.Saver()
        save.save(sess, './vgg/花')
    
    
    



