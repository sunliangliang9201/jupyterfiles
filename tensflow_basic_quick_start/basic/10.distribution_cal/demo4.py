#两台服务器上运行相同的图。这被称为图间复制（in-graph replication）。
import tensorflow as tf
cluster = tf.train.ClusterSpec({"local": ["localhost:2224", "localhost:2225"]})
server1 = tf.train.Server(cluster, job_name="local", task_index=0)
server2 = tf.train.Server(cluster, job_name="local", task_index=1)

graph1 = tf.Graph()
with graph1.as_default():
    var = tf.Variable(0.0, name='var')
sess1 = tf.Session(target=server1.target, graph=graph1)
print(graph1.get_operations())

graph2 = tf.Graph()
sess2 = tf.Session(target=server2.target, graph=graph2)
print(graph2.get_operations())

with graph2.as_default():
    var = tf.Variable(0.0, name='var')
print(graph2.get_operations())

    

