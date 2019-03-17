#如果在所有服务器互联之前尝试在集群上运行某些程序，会发生什么？我们再次创建一个双任务集群。
import tensorflow as tf
from multiprocessing import Process
from time import sleep
#重置tf的图
tf.reset_default_graph()

def myprint(string):
    print("\033[0;31;40m\t%s\033[0m" %(string))
cluster = tf.train.ClusterSpec({
    "local": ["0.0.0.0:4226", "0.0.0.0:4229"]
})
def s1():
    server1 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=0)
    
    with tf.device("/job:local/task:0"):
        no_op = tf.no_op()
        
    sess1 = tf.Session(server1.target)
    for _ in range(6):
        print("Server 1: about to run no-op...", end="")
        sess1.run(no_op)
        print("success!")
        sleep(1.0)

def s2():
    server2 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=1)
    sleep(2.0)
    print("Server 2 dieing...")

#创建进程
p1 = Process(target=s1)
p2 = Process(target=s2)
#进程启动
p1.start()
p2.start()
print("sleep 10","#"*20)
sleep(1)
#重新加入进来的时候会报错
p2 = Process(target=s2, daemon=True)
p2.start()
sleep(10)
#进程中断
p1.terminate()
p2.terminate()
