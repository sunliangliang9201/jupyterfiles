import tensorflow as tf
#查看变量所在的设备
def run_with_location_trace(sess, op):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(op, options=run_options, run_metadata=run_metadata)
    for device in run_metadata.step_stats.dev_stats:
        print(device.device)
        for node in device.node_stats:
            print("  ", node.node_name)
#定义集群服务器
tasks = ["localhost:2222", "localhost:2223"]
jobs = {"local": tasks}
#定义集群变量
cluster = tf.train.ClusterSpec(jobs)
#定义分布服务器
server1 = tf.train.Server(cluster, job_name="local", task_index=0)
server2 = tf.train.Server(cluster, job_name="local", task_index=1)

#重置tf的图
tf.reset_default_graph()
#定义变量
var = tf.Variable(initial_value=0.0, name='var')
#定义两个session
sess1 = tf.Session(server1.target)
sess2 = tf.Session(server2.target)
print(id(sess1.graph))
print(id(sess2.graph))
sess1.run(tf.global_variables_initializer())
#sess2.run(tf.global_variables_initializer())
print("Initial value of var in session 1:", sess1.run(var))
print("Initial value of var in session 2:", sess2.run(var))

sess1.run(var.assign_add(7.0))
print("Incremented var in session 1")
#注意这里的变量
print("Value of var in session 1:", sess1.run(var))
print("Value of var in session 2:", sess2.run(var))
run_with_location_trace(sess1, var)
run_with_location_trace(sess1, var.assign_add(1.0))
print("#"*50)
run_with_location_trace(sess2, var)
#关闭session
sess1.close()
sess2.close()