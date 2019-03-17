import tensorflow as tf
from multiprocessing import Process
from time import sleep

def myprint(string):
    print("\033[0;31;40m\t%s\033[0m" %(string))
cluster = tf.train.ClusterSpec({
    "local": ["localhost:3226", "localhost:3227"]
})
def s1():
    server1 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=0)
    var = tf.Variable(10.0, name='var')
    sess1 = tf.Session(server1.target)
    
    print("Server 1: waiting for connection...")
    sess1.run(tf.report_uninitialized_variables())
    while len(sess1.run(tf.report_uninitialized_variables())) > 0:
        print("Server 1: waiting for initialization...")
        sleep(1.0)
    print("Server 1: variables initialized!")
    print(sess1.run(var))
#S2也加入到以后20秒集群
def s2():
    server2 = tf.train.Server(cluster,
                              job_name="local",
                              task_index=1)
    var = tf.Variable(4.0, name='var')
    sess2 = tf.Session(server2.target)
    
    for i in range(3):
        print("Server 2: waiting %d seconds before initializing..."
              % (3 - i))
        sleep(1.0)
    #print(sess2.run(var))
    sess2.run(tf.global_variables_initializer())
# daemon=True so that these processes will definitely be killed
# when the notebook restarts
#创建进程
p1 = Process(target=s1,daemon=True)
p2 = Process(target=s2,daemon=True)
#进程启动
p1.start()
p2.start()
print("sleep 10","#"*20)
sleep(10)
#重新加入进来的时候会报错
#p2 = Process(target=s2, daemon=True)
#p2.start()
#sleep(10)
#进程中断
p1.terminate()
p2.terminate()
