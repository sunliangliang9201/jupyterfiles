#如果在所有服务器互联之前尝试在集群上运行某些程序，会发生什么？我们再次创建一个双任务集群。
import tensorflow as tf
from multiprocessing import Process
from time import sleep

def myprint(string):
    print("\033[0;31;40m\t%s\033[0m" %(string))
cluster = tf.train.ClusterSpec({
    "local": ["localhost:3226", "localhost:3227"]
})
#以下两个函数都是加入到一个集群里面，S1加入到集群
def s1():
    server1 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=0)
    var = tf.Variable(initial_value=10.0)
    with tf.Session(server1.target) as sess:
        myprint("server 1: running no-op...")
        sess.run(tf.global_variables_initializer())
        myprint(sess.run(var))
        myprint("server 1: no-op run!")
    server1.join() # Block
#等S1加入到集群以后，S2也加入到集群
def s2():  
    for i in range(10):
        myprint("server 2: %d seconds left before connecting..."
              % (10 - i))
        sleep(1.0)
    server2 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=1)
    var2 = tf.Variable(initial_value=11.0)
    with tf.Session(server2.target) as sess:
        myprint("server 2: running no-op...")
        sess.run(tf.global_variables_initializer())
        myprint(sess.run(var2))
        myprint("server 2: no-op run!")
    myprint("server 2: connected!")
    

    server2.join() # Block
   
#创建进程
p1 = Process(target=s1,daemon=True)
p2 = Process(target=s2,daemon=True)
#进程启动
p1.start()
p2.start()
sleep(20)
#进程中断
p1.terminate()
p2.terminate()
