import tensorflow as tf
from Model_Tensorboard import Model
from preprocess import PreProcess
from datetime import datetime
import time
import os
from PIL import Image
import numpy as np
from accuracy import Eval
from inference import Infer

def _train(data, path, checkpoint=None):
	batch_size = 32
	learning_rate = 1e-4
	patience = 10
	ipt = 10
	decay_steps = 10000
	decay_rate = 0.9

	train = PreProcess(data)

	with tf.Graph().as_default():
		image_batch, length_batch, digits_batch = train.return_Data()
		length_logits, digits_logits = Model.inference(image_batch)
		loss = Model.loss(length_logits, digits_logits, length_batch, digits_batch)

		global_step = tf.Variable(0, name='global_step', trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)

		tf.summary.scalar('loss', loss)
		summary = tf.summary.merge_all()

		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter(os.path.join(path, 'summary'), sess.graph)
			sess.run(tf.global_variables_initializer())
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			saver = tf.train.Saver()
			if checkpoint is not None:
				saver.restore(sess, checkpoint)

			print('Start training')
			best_accuracy = 0.0
			duration = 0.0

			i = 0
			while True:
				start_time = time.time()
				_, loss_val, global_val, summary_val = sess.run([train_op, loss, global_step, summary])

				if global_val % 2 == 0:
					print(':::: Steps %d, loss = %f' % (global_val, loss_val))
					summary_writer.add_summary(summary_val, global_step=global_val)

				# Approx 1000 batches for 1 epoch
				if global_val % 5 != 0:
					continue

				print(':: Evaluating on validation dataset...')
				saver.save(sess, checkpoint)
				evalu = Eval()
				accuracy = evalu.accuracy(data,checkpoint)
				print(':::: accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

				if accuracy > best_accuracy:
					path_to_checkpoint_file = saver.save(sess, os.path.join(path, 'best_model.ckpt'))
					patience = ipt
					best_accuracy = accuracy
				else:
					patience -= 1

				print(':: patience = %d' % patience)
				if patience == 0:
					break

			coord.request_stop()
			coord.join(threads)
			print('Finished')


def main(_):

	#test = PreProcess('data/')
	#test.prepare_Data()
	_train('data/','logs/','./latest.ckpt')
	t = Infer('./latest.ckpt')
	#for i in range(1000):
	#	print(t.infer('data/cropped_images/'+ str(i+1) + '.png'))
	print(t.inferi(np.array(Image.open('1.png'))))
	#infer('data/cropped_images/1.png','data/checkpoint2/latest.ckpt')
	#t1 = Eval()
	#print(t1.accuracy('data/','data/latest.ckpt'))


if __name__ == '__main__':
	tf.app.run(main=main)
