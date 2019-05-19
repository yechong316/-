import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imshow, imread, imresize
# ######################################################
# 定义神经网络 在此采用vgg11，但是取消第五层卷积层
# ######################################################

def vgg_network(x, n_class):
    '''设计网络权重、偏置以及所使用这些变量进行的卷积、池化和全连接过程'''
    weight = {
        'wc1_1': tf.Variable(tf.cast(np.random.random([3, 3, 3, 16]   ), dtype=tf.float32), 'wc1_1'),
        'wc2_1': tf.Variable(tf.cast(np.random.random([3, 3, 16, 32]  ), dtype=tf.float32), 'wc2_1'),
        'wc3_1': tf.Variable(tf.cast(np.random.random([3, 3, 32, 64]  ), dtype=tf.float32), 'wc3_1'),
        'wc3_2': tf.Variable(tf.cast(np.random.random([3, 3, 64, 64]  ), dtype=tf.float32), 'wc3_2'),
        'wc4_1': tf.Variable(tf.cast(np.random.random([3, 3, 64, 128] ), dtype=tf.float32), 'wc4_1'),
        'wc4_2': tf.Variable(tf.cast(np.random.random([3, 3, 128, 128]), dtype=tf.float32), 'wc4_2'),
        'wc5_1': tf.Variable(tf.cast(np.random.random([3, 3, 128, 256]), dtype=tf.float32),  'wc5_1'),
        'wc5_2': tf.Variable(tf.cast(np.random.random([3, 3, 256, 256]), dtype=tf.float32), 'wc5_2'),

        # 全连接层
        'wfc_1': tf.Variable(tf.cast(np.random.random([1 * 1 * 256, 256]), dtype=tf.float32), 'wfc_1'),
        'wfc_2': tf.Variable(tf.cast(np.random.random([1 * 1 * 256, 256]), dtype=tf.float32), 'wfc_2'),
        'wfc_3': tf.Variable(tf.cast(np.random.random([256, n_class]), dtype=tf.float32),     'wfc_3'),
    }

    biase = {
        'bc1_1': tf.Variable(tf.cast(np.random.random([16]),dtype=tf.float32),'bc1_1'),

        'bc2_1': tf.Variable(tf.cast(np.random.random([32]),dtype=tf.float32),'bc2_1'),

        'bc3_1': tf.Variable(tf.cast(np.random.random([64]),dtype=tf.float32),'bc3_1'),
        'bc3_2': tf.Variable(tf.cast(np.random.random([64]),dtype=tf.float32),'bc3_2'),

        'bc4_1': tf.Variable(tf.cast(np.random.random([128]), dtype=tf.float32), 'bc4_1'),
        'bc4_2': tf.Variable(tf.cast(np.random.random([128]), dtype=tf.float32), 'bc4_2'),

        'bc5_1': tf.Variable(tf.cast(np.random.random([256]),dtype=tf.float32), 'bc5_1'),
        'bc5_2': tf.Variable(tf.cast(np.random.random([256]),dtype=tf.float32), 'bc5_2'),
        # 全连接层,)
        'bfc_1': tf.Variable(tf.cast(np.random.random([256]),dtype=tf.float32),     'bfc_1'),
        'bfc_2': tf.Variable(tf.cast(np.random.random([256]),dtype=tf.float32),     'bfc_2'),
        'bfc_3': tf.Variable(tf.cast(np.random.random([n_class]),dtype=tf.float32),'bfc_3'),
    }

    #############
    # conv_1
    #############
    # 定义动态卷积、池化、全连接过程(参数如何使用)
    # 卷积过程

    net = tf.nn.conv2d(input=x, filter=weight['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc1_1']))
    # 局部响应归一化
    # net = tf.nn.lrn(net)
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #############
    # conv_2
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc2_1']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #############
    # conv_3
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #############
    # conv_4
    #############
    # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #############
    # conv_5
    #############
    # # 卷积
    net = tf.nn.conv2d(input=net, filter=weight['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_1']))

    net = tf.nn.conv2d(input=net, filter=weight['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_2']))
    # 池化
    net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 构建全连接部分的网络结构
    # 拉升
    net = tf.reshape(net, shape=[-1, weight['wfc_1'].get_shape()[0]])

    ######################
    # fc1
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_1']) + biase['bfc_1'])

    ######################
    # fc2
    ######################
    net = tf.nn.relu(tf.matmul(net, weight['wfc_2']) + biase['bfc_2'])

    ######################
    # fc3(out)
    ######################
    # net = tf.matmul(net, weight['wfc_3']) + biase['bfc_3']

    # 注意：全连接的最后一层输出所使用的激活函数是likelyhood-softmax
    # print('最后一层输出的值为', net[0])
    return tf.matmul(net, weight['wfc_3']) + biase['bfc_3']

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
pred = vgg_network(x, n_class)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

pred_1 = tf.nn.softmax(pred)

######################
#evaluation
######################
acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32), axis = None)

# ######################################################
# 训练模型
# ######################################################
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    epochs = 1000
    batch_size = 256
    display_epoch = 20

    for epoch in range(epochs):
        sess.run(tf.global_variables_initializer())

        cost_, accurate_ = [], []
        batch = data.shape[0] // batch_size
        for i in range(batch):

            x_train = data[batch_size * i : batch_size * (i + 1)]
            y_train = label[batch_size * i : batch_size * (i + 1)]
            cost, accurate, _ = sess.run([loss, acc, opt], feed_dict={x:x_train, y:y_train, lr:1})
            cost_.append(cost)
            accurate_.append(accurate)
            # print('预测值：', predict)
            # cost_total.append(cost)
        # if (epoch) % display_epoch == 0:

        print('epoch:{}, cost:{}, acc:{}'.format(epoch, cost_[0], accurate_[0]))

            # plt.scatter(epoch, cost)
            # plt.ion()
            # plt.pause(0.2)
            # plt.show()
        # if epoch % 10 == 0:

            # print('epochs:{}, cost:{}'.format(epoch, cost))



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



