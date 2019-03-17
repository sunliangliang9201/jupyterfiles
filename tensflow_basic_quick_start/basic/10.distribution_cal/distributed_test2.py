# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
import time
start = time.clock()
train_X = np.random.rand(100).astype(np.float32)
train_Y = train_X * 0.1 + 0.3

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="reminder")
y = w * X + b
loss = tf.reduce_mean(tf.square(y - Y))
init_op = tf.global_variables_initializer()
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 选择创建session使用的master
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(5000):
        sess.run(train_op, feed_dict={X: train_Y, Y: train_Y})
        if i % 50 == 0:
            print(i, sess.run(w), sess.run(b))

    print(sess.run(w))
    print(sess.run(b))
elapsed = (time.clock() - start)
print("Time used:",elapsed)