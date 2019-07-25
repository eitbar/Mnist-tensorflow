from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import myMnist
import input_data

def placeholder_inputs(graph):
	images_placeholder = graph.get_tensor_by_name("img_pl:0")
	labels_placeholder = graph.get_tensor_by_name("lab_pl:0")
	return images_placeholder, labels_placeholder

def fill_feed_dict(images_pl, labels_pl, data_sets):
	ran = int(np.random.uniform(0,1000))
	batch_xs, batch_ys = data_sets.test.next_batch(ran + 1)
	feed_dict = {
		images_pl : batch_xs[[ran]],
		labels_pl : batch_ys[[ran]],
	}
	return feed_dict


def test_save(sess, logits, images_pl, labels_pl, data_sets) :
	feed_dict = fill_feed_dict(images_pl, labels_pl, data_sets)
	result = sess.run(logits, feed_dict=feed_dict)
	print(result)
	print(sess.run(tf.argmax(feed_dict[labels_pl],1)))


def run_test():

	data_sets=input_data.read_data_sets("data/", one_hot=True)
	sess = tf.Session()
	saver = tf.train.import_meta_graph('my_mnist_model.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()
	logits = graph.get_tensor_by_name("logits_for_restore:0")
	images_placeholder , labels_placeholder = placeholder_inputs(graph)
	test_save(sess, logits, images_placeholder, labels_placeholder, data_sets)

def main(_):
	run_test()

if __name__=='__main__':
	tf.app.run()
