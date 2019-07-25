import math
import tensorflow as tf


NUM_CLASSES = 10

IMAGE_SIZE = 28

IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units) :
	with tf.name_scope("hidden1") as scope:
		weights = tf.Variable(
			tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
								stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
			name = "weights")
		biases = tf.Variable(tf.zeros([hidden1_units]), name="biases")
		hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
	
	with tf.name_scope("hidden2") as scope:
		weights = tf.Variable(
			tf.truncated_normal([hidden1_units, hidden2_units],
								stddev = 1.0/math.sqrt(float(hidden1_units))),
			name = "weights")
		biases = tf.Variable(tf.zeros([hidden2_units]), name="biases")
		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
	
	with tf.name_scope("liner") as scope:
		weights = tf.Variable(
				tf.truncated_normal([hidden2_units, NUM_CLASSES],
									stddev = 1.0/math.sqrt(float(hidden2_units))),
				name="weights")
		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name="biases")
		logits = tf.matmul(hidden2, weights) + biases
	return logits

def loss(logits, one_hot_labels) :
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = one_hot_labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
	return loss

def train(loss, learning_rate):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name="global_step", trainable = False)
	train_op = optimizer.minimize(loss, global_step = global_step)
	return train_op

def evaluation(logits, one_hot_label):
	correct = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_label,1))
	eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
	return eval_correct	
