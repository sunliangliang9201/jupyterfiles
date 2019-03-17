#创建本地的服务器进行参数共享
import tensorflow as tf
from time import sleep
var1 = tf.Variable(3.0, name="var1")

# 创建一个本地TensorFlow服务器
server = tf.train.Server.create_local_server()
# 在集群上创建一个会话
sess1 = tf.Session(server.target)
sess2 = tf.Session(server.target)
sess2.run(tf.global_variables_initializer())

print("sess1:",sess1.run(var1))
print("sess2:",sess2.run(var1))
sleep(20)