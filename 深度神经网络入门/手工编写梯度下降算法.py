import numpy as np
import matplotlib.pyplot as plt

def relu(x): return [max(i, 0) for i in x]

x = np.linspace(-5, 5, 50).reshape(-1, 1)

y = 3*x +1 + (np.random.random((50, 1)) - 0.5)/10
y1 = 3*x + 1
plt.plot(x, y, 'r')
plt.plot(x, y1, 'b')
plt.show()

def layer(in_channel, out_channel, bais=True):

    w = np.random.random(size=[in_channel, out_channel])

    if bais:b = np.random.random(size=[out_channel])
    else: b = 0

    return w, b

def MSE(label, target):

    loss = np.square((label - target))
    return np.mean(loss)

def update(label, target, w, b, x, lr=0.01):

    w -= -np.mean(np.multiply((label - target), x)) * lr
    b -= -np.mean((label - target)) * lr

    return w, b


w, b = 1, 0
# w2, b2 = layer(10, 1)

epoch = 100

for i in range(epoch):


    x = w * x + b
    loss = MSE(y, x)
    print('loss', loss)
    w, b = update(label=y, target=x, w=w, b=b, x=x)

    if loss < 0.5:
        break

print(w, b)