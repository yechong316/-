import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imshow, imread, imresize
# ######################################################
# 定义神经网络 在此采用vgg11，但是取消第五层卷积层
# ######################################################

def vgg_network(x, n_class):

    net_total = {}
    '''设计网络权重、偏置以及所使用这些变量进行的卷积、池化和全连接过程'''
    weight = {
        'wc1_1': tf.get_variable('wc1_1', [3, 3, 3, 16]   , dtype=tf.float32),
        'wc2_1': tf.get_variable('wc2_1', [3, 3, 16, 32]  , dtype=tf.float32),
        'wc3_1': tf.get_variable('wc3_1', [3, 3, 32, 64]  , dtype=tf.float32),
        'wc3_2': tf.get_variable('wc3_2', [3, 3, 64, 64]  , dtype=tf.float32),
        'wc4_1': tf.get_variable('wc4_1', [3, 3, 64, 128] , dtype=tf.float32),
        'wc4_2': tf.get_variable('wc4_2', [3, 3, 128, 128], dtype=tf.float32),
        'wc5_1': tf.get_variable('wc5_1', [3, 3, 128, 256], dtype=tf.float32),
        'wc5_2': tf.get_variable('wc5_2', [3, 3, 256, 256], dtype=tf.float32),
        # 
        # # 全连接层
        'wfc_1': tf.get_variable('wfc_1', [1 * 1 * 256, 256], dtype=tf.float32),
        'wfc_2': tf.get_variable('wfc_2', [1 * 1 * 256, 256], dtype=tf.float32),
        'wfc_3': tf.get_variable('wfc_3', [256, n_class],     dtype=tf.float32),
    }

    biase = {
        'bc1_1': tf.get_variable('bc1_1', [16], dtype=tf.float32),

        'bc2_1': tf.get_variable('bc2_1', [32], dtype=tf.float32),

        'bc3_1': tf.get_variable('bc3_1', [64], dtype=tf.float32),
        'bc3_2': tf.get_variable('bc3_2', [64], dtype=tf.float32),

        'bc4_1': tf.get_variable('bc4_1', [128], dtype=tf.float32),
        'bc4_2': tf.get_variable('bc4_2', [128], dtype=tf.float32),

        'bc5_1': tf.get_variable('bc5_1', [256],dtype=tf.float32,),
        'bc5_2': tf.get_variable('bc5_2', [256],dtype=tf.float32,),

        'bfc_1': tf.get_variable('bfc_1', [256],dtype=tf.float32,),
        'bfc_2': tf.get_variable('bfc_2', [256],dtype=tf.float32,),
        'bfc_3': tf.get_variable('bfc_3', [n_class],dtype=tf.float32,),
    }

    #############
    # conv_1
    #############
    # 定义动态卷积、池化、全连接过程(参数如何使用)
    # 卷积过程

    net = tf.nn.conv2d(input=x, filter=weight['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc1_1']))
    # 局部响应归一化
    # net = tf.nn.lrn(net)
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net_total['conv_1'] = net

    #############
    # conv_2
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc2_1']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    net_total['conv_2'] = net
    #############
    # conv_3
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc3_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc3_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    net_total['conv_3'] = net
    #############
    # conv_4
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc4_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc4_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    net_total['conv_4'] = net
    #############
    # conv_5
    #############
    # # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc5_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, biase['bc5_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    net_total['conv_5'] = net
    # 构建全连接部分的网络结构
    # 拉升
    net = tf.reshape(net, shape=[-1, weight['wfc_1'].get_shape()[0]])

    ######################
    # fc1
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_1']) + biase['bfc_1'])
    net_total['fc1'] = net
    ######################
    # fc2
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_2']) + biase['bfc_2'])
    net_total['fc2'] = net
    ######################
    # fc3(out)
    ######################
    # net = tf.matmul(net, weight['wfc_3']) + biase['bfc_3']

    # 注意：全连接的最后一层输出所使用的激活函数是likelyhood-softmax
    # print('最后一层输出的值为', net[0])
    output = tf.matmul(net, weight['wfc_3']) + biase['bfc_3']
    net_total['fc3'] = output
    return output, net_total

# ######################################################
# 加载数据并进行数据预处理
# ######################################################
datasets_path = r"D:\迅雷下载\train_32x32.mat"
mat = sio.loadmat(datasets_path)
data = mat['X']
# print('data的格式为{}:'.format(data.shape))
label_ = mat['y']
# plt.imshow(data[:, :, :, 0])
# plt.show()
# 对数据进行采样
number = 10000
np.random.seed(40)
data = np.transpose(data, [3, 0, 1, 2])
data = data - np.mean(data, axis=(0, 1, 2))
# samples = np.random.choice(data.shape[0], number)
# data_sample = np.array([
#    data[i] for i in samples
# ])
# data_sample = data[samples]
# label = label[samples]
# label_sample = np.zeros(shape=[label.shape[0], 10])
label = np.zeros(shape=[label_.shape[0], 10])
for i in range(label.shape[0]):

    num = label_[i][0]
    if num == 10:

        label[i][0] = 1
    else:

        label[i][num] = 1

# print('data的格式为{}:'.format(data_sample.shape))
# print('label的格式为{}:'.format(label_sample.shape))

# plt.imshow(data_sample[0])
# plt.show()
# ######################################################
# 构建神经网络初始参数
# ######################################################
n_class = 10
lr = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, n_class])

# ######################################################
# 损失函数及优化
# ######################################################
pred, layer_output = vgg_network(x, n_class)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)  # 损失函数
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

pred_1 = tf.nn.softmax(pred)

######################
#evaluation
######################
acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32))

# ######################################################
# 训练模型
# ######################################################
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    epochs = 1000
    batch_size = 1600 * 5
    display_epoch = 20
    base_lr = 0.01
    learning_rate = base_lr
    for epoch in range(epochs):
        sess.run(tf.global_variables_initializer())

        cost_, accurate_ = [], []
        batch = data.shape[0] // batch_size
        for i in range(batch):

            # x_train = data[batch_size * i : batch_size * (i + 1)]
            # y_train = label[batch_size * i : batch_size * (i + 1)]

            X_train = data[i * batch_size:i * batch_size + batch_size]
            Y_train = label[i * batch_size:i * batch_size + batch_size]

            sess.run(opt, feed_dict={x: X_train, y: Y_train, lr: learning_rate})
            cost, accuaray = sess.run([loss, acc], {x: X_train, y: Y_train, lr: learning_rate})
            print('step:%s,loss:%f,acc:%f' % (str(epoch) + '-' + str(i), cost[0], accuaray))
            # 动态修改学习率
            learning_rate = base_lr * (1 - epoch / epochs) ** 2
    save = tf.train.Saver()
    save.save(sess, './resnet/门牌号码')



    # ######################################################
    # 预测图片
    # ######################################################

    # png = r"D:\人工智能\数据集\门牌号码识别\train\29017.png"
    # img = imread(png, mode='RGB')
    #
    # print('预测图片的尺寸为：', img.shape)
    # img = imresize(img, [32, 32, 3])
    # img = np.reshape(img, [1, 32, 32, 3])
    # p = sess.run(pred_1, feed_dict={x:img})
    # print(p)
    # print('分类概率为：'.format(cost))
    # print('分类概率为：'.__dir__(cost))



