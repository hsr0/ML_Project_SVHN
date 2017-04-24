import tensorflow as tf
from Model_Tensorboard import Model
from preprocess import PreProcess
import os

class Eval(object):

	def accuracy(self, path, checkpoint):
		batch_size = 32
		num_batches = 1633 / batch_size
		train = PreProcess(path)

		with tf.Graph().as_default():
			image_batch, length_batch, digits_batch = train.return_val(batch_size)
			length_logits, digits_logits = Model.inference(image_batch)
			length_predictions = tf.argmax(length_logits, axis=1)
			digits_predictions = tf.argmax(digits_logits, axis=2)

			labels = digits_batch
			predictions = digits_predictions

			labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
			predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

			accuracy, update_accuracy = tf.metrics.accuracy(
				labels=labels_string,
				predictions=predictions_string
			)

			tf.summary.scalar('accuracy', accuracy)
			summary = tf.summary.merge_all()

			with tf.Session() as sess:
				summary_writer = tf.summary.FileWriter(os.path.join(path, 'summary'), sess.graph)
				sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess=sess, coord=coord)

				restorer = tf.train.Saver()
				restorer.restore(sess, checkpoint)

				for _ in range(int(num_batches)):
					sess.run(update_accuracy)

				accuracy_val, summary_val = sess.run([accuracy, summary])

				coord.request_stop()
				coord.join(threads)
		return accuracy_val
