import tensorflow as tf
import numpy as np

tf.reset_default_graph()
# 定义分布式集群主机
#PS端
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
#worker端
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# 定义训练服务器的标志
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
def main(_):

    #定义ip和端口 
    ps_hosts = ["0.0.0.0:1980" ]
    worker_hosts = [ "0.0.0.0:1983"]
    #ps_hosts = FLAGS.ps_hosts.split(",")
    #worker_hosts = FLAGS.worker_hosts.split(",")
    

    # ClusterSpec的定义，需要把你要跑这个任务所有的ps和worker的节点的ip和端口信息都包含进去，
    #所有的节点都要执行这段代码，大家就互相知道了，这个集群里都有哪些成员，不同成员的类型是什么，是ps节点还是worker节点
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # tf.train.Server定义开始，每个节点就不一样了。
  #根据执行的命令参数不同，决定了这个任务是哪个任务。
  #如果任务名字是ps的话，程序就join到这里，作为参数更新的服务，等待其他worker节点给他提交参数更新的数据。
  #如果是worker任务，就继续执行后面的计算任务。.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        print("start ps ...")
        server.join()
    elif FLAGS.job_name == "worker":
    # replica_device_setter，在这个with语句之下定义的参数，会自动分配到参数服务器上去定义，
    # 如果有多个参数服务器，就轮流循环分配。
        with tf.device(tf.train.replica_device_setter(
                       worker_device="/job:worker/task:%d" % FLAGS.task_index,
                       cluster=cluster)):
            X = tf.placeholder(tf.float32, [100])
            Y = tf.placeholder(tf.float32, [100])
            # 模型参数
            W = tf.Variable(tf.random_normal([1]), name="weight")
            b = tf.Variable(tf.zeros([1]), name="bias")
            global_step = tf.contrib.framework.get_or_create_global_step()#获得迭代次数
            # 前向结构
            z = tf.multiply(X, W)+ b
            c = X + Y
            tf.summary.histogram('z',z)#将预测值以直方图显示
            #反向优化
            loss =tf.reduce_mean( tf.square(Y - z))
            tf.summary.scalar('loss_function', loss)#将损失以标量显示
            learning_rate = 0.01
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step) #Gradient descent
            saver = tf.train.Saver(max_to_keep=1)
            merged_summary_op = tf.summary.merge_all()#合并所有summary
            init_op = tf.global_variables_initializer()
        #Supervisor，类似于一个监督者，
        #因为分布式了，很多机器都在运行，像参数初始化、保存模型、写summary。
        #这个supervisor帮你一起弄起来了，就不用自己手动去做这些事情了，而且在分布式的环境下涉及到各种参数的共享，
        #其中的过程自己手工写也不好写，于是TensorFlow就给大家包装好这么一个东西。这里的参数is_chief比较重要。
        #is_chief表明了是否为chief supervisor的角色，这里将task_index=0的worker设置成chief superisors
        #logdir是为检查点和summary文件保存的路径
        #init_op表示初始化变量的函数
        #saver需要传入检查点的saver对象传入，superisors会自动保存，如果不想自动保存就设置为None
        #summary_op 也是一样。
        #save_model_secs为保存点的时间间隔。
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=merged_summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=5)

        #连接目标角色创建session
        #这里的session与以往的迭代训练一样，直接进行迭代训练即可，由于使用了supervisor来进行管理session
        #将使用sv.summary_computed函数来保存summary文件，同样，如果想要手动保存检测点文件，也可以使用sv.saver.save
        with sv.managed_session(server.target) as sess:
          # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 1000000:
                train_x = np.random.rand(100).astype(np.float32)
                train_y = train_x * 0.1 + 0.3
                _, step, loss_v, weight, biase = sess.run([optimizer, global_step,loss,W,b],feed_dict={X: train_x, Y: train_y})
                
                if step % 100 == 0:
                    print("step: %d, c: %f" %(step, train_y[0]))
            print("Optimization finished.")
        sv.stop()

if __name__ == "__main__":
    tf.app.run()
