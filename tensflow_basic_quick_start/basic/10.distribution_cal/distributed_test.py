# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
import time
start = time.clock()
train_X = np.random.rand(100).astype(np.float32)
train_Y = train_X * 0.1 + 0.3

# 选择变量存储位置和op执行位置，这里全部放在worker的第1个task上
# 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备
# tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，
# 而计算分配到当前的计算服务器上
# Between-graph replication
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="reminder")
    y = w * X + b
    loss = tf.reduce_mean(tf.square(y - Y))

    init_op = tf.global_variables_initializer()
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 选择创建session使用的master
with tf.Session("grpc://0.0.0.0:1983") as sess:
    sess.run(init_op)
    for i in range(5000):
        sess.run(train_op, feed_dict={X: train_Y, Y: train_Y})
        if i % 50 == 0:
            print(i, sess.run(w), sess.run(b))

    print(sess.run(w))
    print(sess.run(b))
elapsed = (time.clock() - start)
print("Time used:",elapsed)