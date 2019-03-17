import tensorflow as tf
import numpy as np

#生成训练数据
train_X = np.random.rand(100).astype(np.float32)
train_Y = train_X * 0.1 + 0.3

#配置集群
ps_hosts = [ "0.0.0.0:1981", "0.0.0.0:1980" ]
worker_hosts = [ "0.0.0.0:1983", "0.0.0.0:1984", "0.0.0.0:1985" ]
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    tf.app.run()