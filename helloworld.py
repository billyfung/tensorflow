import tensorflow as tf 

#on core2duo laptop, set cores to 2
sess = tf.Session( config=tf.ConfigProto(inter_op_parallelism_threads=2,
                  						 intra_op_parallelism_threads=2))

hello = tf.constant('Hello World')


print sess.run(hello)

a = tf.constant(50)
b = tf.constant(45)

print sess.run(a*b)