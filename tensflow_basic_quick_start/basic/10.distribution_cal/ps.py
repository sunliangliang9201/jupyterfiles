# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Module pyplot not")


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
try:
    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.legend()
    plt.show()
except NameError:
    print("plt is not define")

tf.reset_default_graph()

#定义ip和端口 
strps_hosts="localhost:1681"
strworker_hosts="localhost:1682,localhost:1683,localhost:1684"

#定义角色名称
strjob_name = "ps"
task_index = 0


ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
#集群 (Cluster) : 一个 TensorFlow 的集群里包含了一个或多个作业 (job), 每一个作业又可以拆分成一个或多个任务 (task)。
#集群的概念主要用与一个特定的高层次对象中，比如说训练神经网络，并行化操作多台机器等等。
#集群对象可以通过 tf.train.ClusterSpec 来定义
#参数:1个ps,2个worker
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})
#创建server
server = tf.train.Server(
                    {'ps': ps_hosts,'worker': worker_hosts},
                    job_name=strjob_name,
                    task_index=task_index)

#ps角色使用join进行等待,server.join()对线程进行挂起，开始接收连接消息
if strjob_name == 'ps':
  print("wait")
  server.join()
#在创建网络结构时，使用tf.device函数将全部的节点都放在当前的任务下
#在tf.device函数中的任务是通过tf.train.replica_device_setter来指定
#在tf.train.replica_device_setter中使用worker_device来定义具体的任务名称
#使用cluster的配置来指定角色及对应的IP地址，从而实现管理整个任务的图节点
with tf.device(tf.train.replica_device_setter(
               worker_device="/job:worker/task:%d" % task_index,
               cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    
    global_step = tf.contrib.framework.get_or_create_global_step()#获得迭代次数
    
    # 前向结构
    z = tf.multiply(X, W)+ b
    tf.summary.histogram('z',z)#将预测值以直方图显示
    #反向优化
    cost =tf.reduce_mean( tf.square(Y - z))
    tf.summary.scalar('loss_function', cost)#将损失以标量显示
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step) #Gradient descent

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all()#合并所有summary
   
    init = tf.global_variables_initializer()



#参数设置
training_epochs = 2200
display_step = 2


#创建Supervisor，管理Session
#is_chief表明了是否为chief supervisor的角色，这里将task_index=0的worker设置成chief superisors
#logdir是为检查点和summary文件保存的路径
#init_op表示初始化变量的函数
#saver需要传入检查点的saver对象传入，superisors会自动保存，如果不想自动保存就设置为None
#summary_op 也是一样。
#save_model_secs为保存点的时间间隔。
sv = tf.train.Supervisor(is_chief=(task_index == 0),
                             logdir="log/super/",
                             init_op=init,
                             summary_op=None,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=5)

#连接目标角色创建session
#
#这里的session与以往的迭代训练一样，直接进行迭代训练即可，由于使用了supervisor来进行管理session
#将使用sv.summary_computed函数来保存summary文件，同样，如果想要手动保存检测点文件，也可以使用sv.saver.save
with sv.managed_session(server.target) as sess:
    print("start training...")
    print(global_step.eval(session=sess))
    
    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
    
        for (x, y) in zip(train_X, train_Y):
            _, epoch = sess.run([optimizer,global_step] ,feed_dict={X: x, Y: y})
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X: x, Y: y});
            #将summary 写入文件
            sv.summary_computed(sess, summary_str,global_step=epoch)
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
                if not (np.isnan(loss) ):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
                #sv.saver.save(sess,"log/mnist_with_summaries/",global_step=epoch)
          
    print (" Finished!")
    sv.saver.save(sess,"log/mnist_with_summaries/"+"sv.cpk",global_step=epoch)

sv.stop() 
