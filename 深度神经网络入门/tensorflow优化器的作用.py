'''
y = 2 * x1 + 5

'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
samples = 100
# print(Y)
# for i in range(len(Y)):
#     print(Y[i])
x = np.linspace(-10, 10, 100).reshape(-1, 2) # (50, 2)
y = np.array([2 * i[0] + 3 * i[1] for i in x ]).reshape(-1, 1) # (50,)
layer_1 = 10
layer_2 = 1
# print(x.shape)
# print(y.shape)

with tf.name_scope('fc-1') as fc_1:
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    y_pred = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    w = tf.Variable(tf.random_normal([2, layer_1],mean=-10,
                  stddev=10))
    b = tf.Variable(tf.random_normal([layer_1]))
    h = tf.nn.relu(tf.matmul(input_data, w) + b)

with tf.name_scope('fc-1') as fc_2:

    w = tf.Variable(tf.random_normal([layer_1, layer_2],mean=-10, stddev=10))
    b = tf.Variable(tf.random_normal([layer_2]))
    h = tf.matmul(h, w) + b

    loss = tf.nn.softmax_cross_entropy_with_logits()
    opt = tf.train.AdamOptimizer(1).minimize(loss)

# with tf.name_scope('output') as output:
#     w = tf.Variable(tf.random_normal([1, n_hidden]))
#     b = tf.Variable(tf.random_normal([1]))
#     out = tf.nn.relu(tf.matmul(w, h1) + b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(120):
        weight, bias = sess.run([w, b])

        _, cost = sess.run([opt, loss], feed_dict={input_data:x, y_pred:y})
        print('epochs:{}, w:{}, bias:{}, loss:{}'.format(i + 1, weight, bias, cost.shape))

        # print(cost)
        # h = sess.run(h1, feed_dict={input_data:x})
        # print('h', h)
        # print('h.shape', h.shape)
        # plt.scatter(i, cost)
        # plt.scatter(i + 1, bias)
        # y2 = 2 * np.ones(shape=[120, 1])
        # y3 = 5 * np.ones(shape=[120, 1])
        # plt.plot(list(range(120)), y2, 'r')
        # plt.plot(list(range(120)), y3, 'r')
        # plt.ion()
        # plt.pause(0.1)
        # plt.legend('upper right')
        plt.show()