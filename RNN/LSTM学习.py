import numpy as np
import tensorflow as tf
import random
import collections

# ##########################################r#####
# 构建正反向字典
# ###############################################
with open(r'D:\人工智能\数据集\aclImdb\test\pos\0_10.txt') as f:
    with open(r'D:\人工智能\数据集\aclImdb\test\pos\0_10.txt') as f:
        content = f.read()
        assert f != [], '文本内容为空'

    words = content.split()
    count = collections.Counter(words).most_common()

    dictionary = {}
    for i, _ in count:
        dictionary[i] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

# ##########################################r#####
# 参数定义
# ###############################################
n_input = 3
n_hidden = 128
vocab_size = len(words)
crop = len(dictionary)
weight = tf.get_variable('weight_out', [2 * n_hidden, vocab_size])
bias = tf.get_variable('bias_out', [vocab_size])
# ##########################################r#####
# 随机采样
# ###############################################
def random_samples(offset):
    if offset > crop - n_input - 1:
        offset = random.randint(0, crop - n_input - 1)

    symbols_in_key = [
        [dictionary[str[words[i]]]] for i in range(offset, offset + n_input)
    ]
    symbols_out_key = np.zeros([vocab_size], dtype=float)
    symbols_out_key[dictionary[str(words[offset + n_input])]] = 1
    return symbols_in_key, symbols_out_key


# ##########################################r#####
# 定义LSTM模型
# ###############################################
# LSTM模型的建立
def LSTM(x, weight, bias):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input)

    rnn_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True,
                                           forget_bias=1.0)
    rnn_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True,
                                           forget_bias=1.0)
    output, _, _ = tf.nn.static_bidirectional_rnn(rnn_fw, rnn_bw, x)
    return tf.matmul(output[-1], weight) + bias
x = tf.placeholder(tf.float32, [None, n_input, 1])
y = tf.placeholder(tf.float32, [None, vocab_size])

pred = LSTM(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)

# ##########################################r#####
# 模型评估
# ###############################################
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# ##########################################r#####
# 开始训练
# ###############################################
batch = 5000
batch_size = 20
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(batch):
        x_train, y_train = [], []
        for j in range(batch_size):
            new_x, new_y = random_samples(random.randint(0, crop))
            x_train.append(new_x)
            y_train.append(new_y)

        sess.run(optimizer, feed_dict={x:np.array(x_train), y:np.array(y_train)})

        if (i + 1) % 100 == 0:
            acc, out_pred = sess.run([accuary, pred],
                                     feed_dict={x:np.array(x_train), y:np.array(y_train)})


            symbols_in = [reverse_dictionary[word_index] for word_index in x_train[0]]
            symbols_out = [reverse_dictionary[word_index] for word_index in x_train[0]]


    pass


