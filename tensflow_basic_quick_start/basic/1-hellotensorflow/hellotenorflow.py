import tensorflow as tf

if __name__ == "__main__":
    #定义一个模型
    hello = tf.constant('Hello, TensorFlow!')#定义一个常量
    #运行模型
    sess = tf.Session()
    print(sess.run(hello))
    sess.close()
    

