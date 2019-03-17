#import data
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import tensorflow as tf
import sys
def main(data_dir):
    #数据源的读取
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # 设置模型（这里就是定义学习参数）
    #x是通过placeholder创建的一个占位符，数据的类型是float32类型，None代表可以输入任意张mnist图片，每张图片展开成28x28=784维的向量。
    #   模型同时也需要权值和偏值，这里采用Variable，它代表了一个可以修改的张量。
    #    y=tf.matmul(x, W) + b就代表了
    #    y=Wx+b
    #    可以发现y也会是一个10维的向量，1~10维度，分别代表了这个数字是0~9的概率大小，概率最大者就代表这个手写体被预测为什么
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    #     在机器学习中，需要定义一个函数，来检测模型的好坏。
    #     这个函数称之为cost function，在训练的目的就是最小化这个函数。
    #     这里采用了交叉熵(cross-entropy) 来作为cost function。
    #首先创建一个新的占位符，用来输入正确的值。
    y_ = tf.placeholder(tf.float32, [None, 10])
    #softmax交叉熵
    #然后通过tf.nn.softmax_cross_entropy_with_logits来计算交叉熵
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # 梯度下降法
    #通过梯度下降法来，以0.5的学习速率来寻找交叉熵的局部最小值。
    learning_rate = 0.01 #学习率
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    #初始化变量，创建session
    #通过sess = tf.InteractiveSession()创建了一个交互式的session，
    #如在shell中。然后通过tf.global_variables_initializer().run()来初始化之前所声明的变量。
    #那什么是session呢？
    #    有几点关于tensorflow，是我们所需要知道的。
    #     使用图 (graph) 来表示计算任务.
    #     在被称之为 会话 (Session) 的上下文 (context) 中执行图.
    #     使用 tensor 表示数据.
    #      通过 变量 (Variable) 维护状态.
    #     使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.
    #    所以，我们的计算都是要在session这样子一个上下文环境中来进行的。
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # 训练
    #训练中所需要的参数
    batch_size = 100 #一次获取多少数据进行训练
    total_batch = int(mnist.train.num_examples/batch_size) #模型训练的循环次数
    training_epoch = 100 #整个模型训练多少次
    display_step = 1 #多少次显示一次参数信息
    
    for epoch in range(training_epoch):
        avg_cost = 0
        #开始训练模型，让模型循环训练1000次，每一次随机抽取100个数据来进行训练，
        #然后回到train_step通过梯度下降的方法来进行训练。这称之为随机训练（stochastic training）。
        #这样子每次训练可以使用不同的数据集，降低计算开销，又学习到数据集合的总体数据特征。
        for _ in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            avg_cost += c /total_batch #每一次循环完模型得到一个loss的平均值
        if (((epoch + 1) % display_step) == 0) and (not (avg_cost == "nan")):
            print("Epoch:", '%04d' %(epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Training Finished")
         
        
    # 模型测试
    #通过argmax(y, 1)可以找到y中最大值的下标，然后通过比较y_和y的下标是否相同，
    #如果相同就是正确的为1否则为0，然后把布尔值当成浮点数，进行取平均值，得到一个预测的准确率。
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
