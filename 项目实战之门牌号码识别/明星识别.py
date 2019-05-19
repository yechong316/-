from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import warnings
import os
# from vgg16_python3 import vgg16

warnings.filterwarnings("ignore")
def get_dir(path):  # 获取目录路径
    # 遍历path,进入每个目录都调用visit函数，
    # 有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
    file_paths = []
    for root, dirs, files in os.walk(path):  # 遍历path,进入每个目录都调用visit函数，，有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        for file in files:
            # print(dir)             #文件夹名
            file_paths.append(os.path.join(root, file))  # 把目录和文件名合成一个路径
            # print(dir)             #文件夹名
    return file_paths

def access_classes(datasets_path):
    class_name = []
    file_path = []
    for root, dirs, files in os.walk(datasets_path):
        for dir in dirs:
            # print(dir)             #文件夹名
            class_name.append(dir)
    return class_name

def load_datasets(folder_name):
    images, labels = [], []
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean = 0
    for i in range(len(class_name)):
        image_floder = folder_name + '/' + class_name[i]
        images_path = get_dir(image_floder)
        for j in images_path:
            image_o = np.expand_dims(imresize(imread(j, mode='RGB'), (224, 224)), axis=0)
            image = image_o - mean
            images.append(image)

            label = np.zeros(shape=len(class_name))
            label[i] = 1
            labels.append(np.expand_dims(label, axis=0))

    images_data = np.array(np.concatenate(images, axis=0))
    labels_data = np.array(np.concatenate(labels, axis=0))



    return images_data, labels_data

class vgg11:
    
    def __init__(self, n_class):
        self.weight = {
                'wc1_1': tf.get_variable('wc1_1', [3, 3, 3, 64]),
                'wc2_1': tf.get_variable('wc2_1', [3, 3, 64, 128]),
                'wc3_1': tf.get_variable('wc3_1', [3, 3, 128, 256]),
                'wc3_2': tf.get_variable('wc3_2', [3, 3, 256, 256]),
                'wc4_1': tf.get_variable('wc4_1', [3, 3, 256, 512]),
                'wc4_2': tf.get_variable('wc4_2', [3, 3, 512, 512]),
                'wc5_1': tf.get_variable('wc5_1', [3, 3, 512, 512]),
                'wc5_2': tf.get_variable('wc5_2', [3, 3, 512, 512]),

                'wfc_1': tf.get_variable('wfc_1', [7 * 7 * 512, 4096]),
                'wfc_2': tf.get_variable('wfc_2', [4096, 4096]),
                'wfc_3': tf.get_variable('wfc_3', [4096, n_class])
            }

        self.biase = {
                'bc1_1': tf.get_variable('bc1_1', [64]),
                'bc2_1': tf.get_variable('bc2_1', [128]),
                'bc3_1': tf.get_variable('bc3_1', [256]),
                'bc3_2': tf.get_variable('bc3_2', [256]),
                'bc4_1': tf.get_variable('bc4_1', [512]),
                'bc4_2': tf.get_variable('bc4_2', [512]),
                'bc5_1': tf.get_variable('bc5_1', [512]),
                'bc5_2': tf.get_variable('bc5_2', [512]),
                'bfc_1': tf.get_variable('bfc_1', [4096]),
                'bfc_2': tf.get_variable('bfc_2', [4096]),
                'bfc_3': tf.get_variable('bfc_3', [n_class])
            }
        
        
    def vgg_network(self, image_data):

        #############
        # conv_1
        #############
        # 定义动态卷积、池化、全连接过程(参数如何使用)
        # 卷积过程
        image_x  = tf.cast(x=image_data, dtype=tf.float32)
        net = tf.nn.conv2d(input=image_x, filter=self.weight['wc1_1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc1_1']))
        # 局部响应归一化
        net = tf.nn.lrn(net)
        # 池化
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
        #############
        # conv_2
        #############
        # 卷积
        net = tf.nn.conv2d(input=net, filter=self.weight['wc2_1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc2_1']))
        # 池化
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        #############
        # conv_3
        #############
        # 卷积
        net = tf.nn.conv2d(input=net, filter=self.weight['wc3_1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc3_1']))
        net = tf.nn.conv2d(input=net, filter=self.weight['wc3_2'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc3_2']))
        # 池化
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        #############
        # conv_4
        #############
        # 卷积
        net = tf.nn.conv2d(input=net, filter=self.weight['wc4_1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc4_1']))
        net = tf.nn.conv2d(input=net, filter=self.weight['wc4_2'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc4_2']))
        # 池化
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        #############
        # conv_5
        #############
        # 卷积
        net = tf.nn.conv2d(input=net, filter=self.weight['wc5_1'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc5_1']))
        net = tf.nn.conv2d(input=net, filter=self.weight['wc5_2'], strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.leaky_relu(tf.nn.bias_add(net, self.biase['bc5_2']))
        # 池化
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        # 构建全连接部分的网络结构
        # 拉升
        net = tf.reshape(net, shape=[-1, self.weight['wfc_1'].get_shape()[0]])
    
        ######################
        # fc1
        ######################
        net = tf.nn.relu(tf.matmul(net, self.weight['wfc_1']) + self.biase['bfc_1'])
    
        ######################
        # fc2
        ######################
        net = tf.nn.relu(tf.matmul(net, self.weight['wfc_2']) + self.biase['bfc_2'])
    
        ######################
        # fc3(out)
        ######################
        # 注意：全连接的最后一层输出所使用的激活函数是likelyhood-softmax
        pred = tf.matmul(net, self.weight['wfc_3']) + self.biase['bfc_3']
        return pred

    def load_cpkt(self, png, sess=None):
        del self.weight, self.biase
        self.weight = {
            'wc1_1': tf.Variable(sess.run('wc1_1:{}'.format(0))),
            'wc2_1': tf.Variable(sess.run('wc2_1:{}'.format(0))),
            'wc3_1': tf.Variable(sess.run('wc3_1:{}'.format(0))),
            'wc3_2': tf.Variable(sess.run('wc3_2:{}'.format(0))),
            'wc4_1': tf.Variable(sess.run('wc4_1:{}'.format(0))),
            'wc4_2': tf.Variable(sess.run('wc4_2:{}'.format(0))),
            'wc5_1': tf.Variable(sess.run('wc5_1:{}'.format(0))),
            'wc5_2': tf.Variable(sess.run('wc5_2:{}'.format(0))),
            'wfc_1': tf.Variable(sess.run('wfc_1:{}'.format(0))),
            'wfc_2': tf.Variable(sess.run('wfc_2:{}'.format(0))),
            'wfc_3': tf.Variable(sess.run('wfc_3:{}'.format(0))),
        }

        self.biase = {
            'bc1_1': tf.Variable(sess.run('bc1_1:{}'.format(0))),
            'bc2_1': tf.Variable(sess.run('bc2_1:{}'.format(0))),
            'bc3_1': tf.Variable(sess.run('bc3_1:{}'.format(0))),
            'bc3_2': tf.Variable(sess.run('bc3_2:{}'.format(0))),
            'bc4_1': tf.Variable(sess.run('bc4_1:{}'.format(0))),
            'bc4_2': tf.Variable(sess.run('bc4_2:{}'.format(0))),
            'bc5_1': tf.Variable(sess.run('bc5_1:{}'.format(0))),
            'bc5_2': tf.Variable(sess.run('bc5_2:{}'.format(0))),
            'bfc_1': tf.Variable(sess.run('bfc_1:{}'.format(0))),
            'bfc_2': tf.Variable(sess.run('bfc_2:{}'.format(0))),
            'bfc_3': tf.Variable(sess.run('bfc_3:{}'.format(0)))
        }
        return self.weight, self.biase


if __name__ == '__main__':
    folder_name = "D:\搜狗高速下载\图片\图片\数据集"
    class_name = access_classes(folder_name)
    print(class_name)

    X, Y = load_datasets(folder_name)
    #查看数据的信息
    print(X.shape)
    print(Y.shape)

    #构建与训练相关的参数
    # trainEpoch
    train_epoch = 40
    batch_size = 16 #一般来说这个值是16的整数倍
    display_epoch = 100

    #构建vgg网络所需要的参数及网络结构
    n_class = Y.shape[1]
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None , n_class])
    # 设计损失函数及优化器， 并且建立评估函数
    ######################
    # Loss Function & Optimizer
    ######################
    pred = vgg_network(x=x, n_class=len(class_name))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    ######################
    # evaluation
    ######################
    acc_tf = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc_tf, tf.float32), axis=None)

    ######################
    # train—beginning
    ######################

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                          allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        learning_rate = 0.01
        cost_last = 1e4
        for epoch in range(train_epoch):

            total_batch = X.shape[0] // batch_size  # 104
            for i in range(total_batch):
                X_train = X[i * batch_size: i * batch_size + batch_size]
                Y_train = Y[i * batch_size: i * batch_size + batch_size]
                # print('Epochs:{}, batch:{}'.format(epoch + 1, i))
                sess.run(opt, feed_dict={x: X_train, y: Y_train, lr: learning_rate})

                ######################
            # evaluation
            ######################

            cost, accuaray = sess.run([loss, acc], feed_dict={
                    x: X_train, y: Y_train})
            print('Epochs:{}, loss:{}, acc:{}'.format(
                    epoch + 1, cost[0], accuaray))
            ######################
            # learning_rate change
            ######################
            learning_rate = 0.01 * (1 - epoch / train_epoch) ** 2

        ############################################
        # 保存训练结果
        ############################################
        save = tf.train.Saver()
        save.save(sess, './result/明星识别')
