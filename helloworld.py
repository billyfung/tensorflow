import tensorflow as tf 
import numpy as np

#on core2duo laptop, set cores to 2
sess = tf.Session( config=tf.ConfigProto(inter_op_parallelism_threads=2,
                  						 intra_op_parallelism_threads=2))

hello = tf.constant('Hello World')


print sess.run(hello)

with sess:
	input1 = tf.constant(1, shape = [4])
	input2 = tf.constant(2, shape = [4])
	output = (input1 + input2)
	result = output.eval()
	print result

	input_features = tf.constant(np.reshape([1, 0, 0, 1], (1,4)).astype(np.float32))
	weights = tf.constant(np.random.randn(4,2).astype(np.float32))
	output = tf.matmul(input_features, weights) #matrix multiplication
	print "Input:" 
	print input_features.eval()
	print "Weights:"
	print weights.eval()
	print "Output"
	print output.eval()
