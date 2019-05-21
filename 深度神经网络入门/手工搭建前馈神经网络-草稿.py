import numpy as np

class Nernul_network:

    def __init__(self):

        '''
        输入一个各层的神经元数量，创建W, 和b
        :param n_hiddens:
        '''
        self.length = 0

        self.layer = [{}]



    def add_layer(self, n_hidden, f, x=None):

        '''
        初始化该层的
        :param n_hidden:
        :return:
        '''
        self.length += 1
        if self.length == 1:
            assert x.all() != None, '首次添加隐藏层，必须输入x，否则无法创建！'
            self.layer[0].update({'output':x})
            self.layer.append(
                {'weight': np.random.normal(size=[x.shape[1], n_hidden]),
                 'bias': np.random.normal(size=[n_hidden]),
                 'active': f,
                 'dim': n_hidden,
                 'hidden': None,
                 'output': str
                 })

        else:
            pior_dim = self.layer[self.length - 1]['dim']
            self.layer.append(
            {'weight': np.random.normal(size=[pior_dim, n_hidden]),
            'bias': np.random.normal(size=[n_hidden]),
            'active': f,
            'dim': n_hidden,
            'hidden': None,
            'output': str,
             }
        )


    def train(self):


        for i in range(1, self.length + 1):

            x_ = self.layer[i - 1]['output']
            weight = self.layer[i]['weight']
            bias = self.layer[i]['bias']
            fun = self.layer[i]['active']

            h = np.dot(x_, weight) + bias
            y = fun(h)
            self.layer[i]['hidden'] = h
            self.layer[i]['output'] = y


    def lost(self, labels):

        assert self.layer[-1]['output'].any() != str, '请执行Nernul_network中的train再执行此方法！'
        self.labels = labels
        self.logits = self.layer[self.length]['output']
        lost = np.mean(np.square(self.labels - self.logits))
        return lost

    def optimize(self, lr):

        dl_dy = np.mat(-(self.labels - self.logits))
        h_ = np.mat(self.layer[-1]['hidden'])
        y_ = np.mat(self.layer[-1]['output'])
        w_ = self.layer[-1]['weight']

        dy_dw =  y_.T * np.multiply(dl_dy, np.multiply(h_, 1 - h_))
        w_ -= lr * dy_dw
        '''
        这里需要有一个遍历的过程，留待后续解决
        '''
        # for i in range(self.length - 1, 0, -1):
        #
        #     y_n_1 = y_
        #     h_ = self.layer[i]['hidden']
        #     y_ = self.layer[i]['output']
        #     w_ = self.layer[i]['weight']
        #     dy_dw *= h_ * (1 - h_) * y_ / y_n_1
        #     dy_dw *= y_.T * dy_dw
        #
        #     w_ -= lr * dy_dw

        dL_dY = np.multiply((self.labels - self.logits), self.layer[-1]['hidden'])







def sigoid(x): return  1 / ( 1 + np.exp(-x))

def divsigoid(h): return  h * ( 1 - h )



input = np.random.random(size=[2, 2])
y_true = input ** 2 + 0.3 * np.random.randint(0, 1)

nn = Nernul_network()
layer_1 = nn.add_layer(2, sigoid, x=input)
layer_2 = nn.add_layer(1, sigoid)


nn.train()
lost1 = nn.lost(y_true)

for i in range(len(nn.layer)):
    print(input)
    print('隐藏层-{} 输出的值：{}'.format(i, nn.layer[i]['output']))

print('lost:', lost1)
# epochs = 100
# for i in range(epochs):


    # pass
    # opitimizer(lost)
    # print(w1, w2)
