'''
照着深度学习老师的做法,复现一次RNN网络
'''

import tensorflow as tf
import numpy as np
import collections

filename = 'test.txt'

with open(filename) as f:
    content = f.read()

words = content.split()
print(words)