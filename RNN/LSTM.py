'''
本脚本存放一个LSTM的方法包,供其他文件进行调用
'''

import collections
import tensorflow as tf
import random
import numpy as np
import sys

class LSTM:

    # ###############################
    # 构建正反向字典
    # ###############################

    def constructed_dict(words):
        connt = collections.Counter(words).most_common()
        # [('to', 11), ('the', 8), ('I', 6), ('a', 6), ('can', 5),
        dict_word = dict()
        for word, _ in connt:
            dict_word[word] = len(dict_word)

        # 反向字典
        reverse_dict_word = dict(zip(dict_word.values(),
                                     dict_word.keys()))

        return dict_word, reverse_dict_word

    # LSTM模型的建立
    def LSTM(x, weight, bias):
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(x, n_input, 1)

        rnn_cell_forward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True
                                                        , forget_bias=1.0)  # 当state_is_tuple=False# 时,变成GRU
        rnn_cell_backward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True
                                                         , forget_bias=1.0)  # 当state_is_tuple=False

        # output 从隐藏层传入的
        output, output_states_fw, output_state_bw = tf.nn.static_bidirectional_rnn(rnn_cell_forward, rnn_cell_backward,
                                                                                   x, dtype=tf.float32)

        return tf.matmul(output[-1], weight) + bias



    def word_random_sample(self, filename, offset):
        # # ############################
        # 语料读取  1G以上的数语料才能有商业化的价值
        # ############################
        content = ''
        with open(filename, 'r') as f:
            content = f.read()
        words = content.split()
        crop_size = len(words)

        while offset + n_input > crop_size:
            random.randint(0, crop_size - n_input)

        symbols_in_key = [[dict[str(words[i])]]
                          for i in range(offset, offset + n_input)]

        # y值的确定
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)

        symbols_out_onehot[dict[str(words[offset + n_input])]] = 1

        return symbols_in_key, symbols_out_onehot

    # ######################################
    # 构建符合 TensorFlow计算的形参类型和损失函数
    # ######################################
    x = tf.placeholder(tf.float32, [None, n_input, 1])
    y = tf.placeholder(tf.float32, [None, vocab_size])

    predict = LSTM(x, weight, bias)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)

    # ######################################
    # 精度
    # ######################################
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accary = tf.reduce_mean(tf.cast(correct, tf.float32))

    # with tf.Session as sess:
    # with tf.device('/gpu:1'):
    # with tf.device('/GPU:1'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):  # 一般都是两个for
            x_train, y_train = [], []
            for j in range(batch_size):  # 随机采样, 和随机复采样不一样
                new_x, new_y = word_random_sample(random.randint(0, crop_size))
                x_train.append(new_x)
                y_train.append(new_y)

            sess.run(optimizer, feed_dict={x: x_train, y: y_train})

            if i % 100 == 0:
                acc, out_pred = sess.run([accary, predict], feed_dict={x: x_train, y: y_train})

                # ####################
                # 用正反字典输入和输出
                # ####################

                symbols_in = [rever_dict[_[0]] for _ in x_train[0]]

                symbols_out = np.array(int(np.argmax(y_train, 1)[0]))

                pred_out = rever_dict[int(np.argmax(out_pred, 1)[0])]

                print('精确率:%f' % acc)
                # print('%s - [%s] vs [%s]'%(symbols_in,symbols_out))
                print('%s-[%s]vs[%s]' % (symbols_in, symbols_out, pred_out))






