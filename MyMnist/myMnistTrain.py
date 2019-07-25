import time
import tensorflow as tf
import numpy
import input_data
import myMnist

settings = {
	"learning_rate" : 0.01,
	"max_steps" : 4000,
	"hidden1" : 128,
	"hidden2" : 32,
	"batch_size" : 100,
}

def placeholder_inputs():
	images_placeholder = tf.placeholder(tf.float32, shape=(None, myMnist.IMAGE_PIXELS),name="img_pl")
	labels_placeholder = tf.placeholder(tf.int32, shape=(None, myMnist.NUM_CLASSES),name="lab_pl")
	return images_placeholder, labels_placeholder

def fill_feed_dict(data_sets, images_pl, labels_pl, batch_size):
	images_feed, labels_feed = data_sets.next_batch(batch_size)
	feed_dict = {
		images_pl : images_feed,
		labels_pl : labels_feed, 
	}
	return feed_dict

def do_eval(sess, eval_correct, images_pl, labels_pl, data_sets, batch_size) :
	true_count = 0
	num_batch = int(data_sets.num_examples / batch_size)
	num_examples = num_batch * batch_size
	for i in range(num_batch):
		feed_dict = fill_feed_dict(data_sets, images_pl, labels_pl, batch_size)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	precision = float(true_count) / float(num_examples)
	print('Num examples: %d  Num correct: %d  Precision: %0.04f' %
		(num_examples, true_count, precision))
		

def run_training():
	batch_size = settings["batch_size"]
	hidden1 = settings["hidden1"]
	hidden2 = settings["hidden2"]
	learning_rate = settings["learning_rate"]
	max_steps = settings["max_steps"]	

	data_sets = input_data.read_data_sets("data/",one_hot=True)
	
	with tf.Graph().as_default():
		images_placeholder , labels_placeholder = placeholder_inputs()	
		logits = myMnist.inference(images_placeholder, hidden1, hidden2)
		loss = myMnist.loss(logits, labels_placeholder)
		train_op = myMnist.train(loss, learning_rate)
		logits_result_for_restore = tf.argmax(logits,1,name="logits_for_restore")
		eval_correct = myMnist.evaluation(logits, labels_placeholder)

		init = tf.initialize_all_variables()
		
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)	
		for step in range(max_steps) :
			start_time = time.time()
			feed_dict = fill_feed_dict(data_sets.train,
										images_placeholder,
										labels_placeholder,
										batch_size)
			_, loss_value = sess.run([train_op, loss],
									feed_dict=feed_dict)
			duration = time.time() - start_time
			if step % 100 == 0 :
				print ("step %d: loss = %.5f (use %.5f sec)" % (step, loss_value, duration))
				
			if (step+1) % 1000 == 0 :
				print("train_set:")
				do_eval(sess,
						eval_correct,
						images_placeholder,
						labels_placeholder,
						data_sets.train,
						batch_size)
				print("Validation_set:")
				do_eval(sess,
						eval_correct,
						images_placeholder,
						labels_placeholder,
						data_sets.validation,
						batch_size)
				print("Test_set:")
				do_eval(sess,
						eval_correct,
						images_placeholder,
						labels_placeholder,
						data_sets.test,
						batch_size)
			if (step+1) == max_steps :
				saver.save(sess, 'my_mnist_model')			

def main(_):
	run_training()

if __name__=='__main__':
	tf.app.run()
