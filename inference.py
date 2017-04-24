import tensorflow as tf
from Model_Tensorboard import Model
from preprocess import PreProcess
from PIL import Image
import numpy as np
import os

class Infer(object):

	def __init__ (self, checkpoint):
		self.checkpoint = checkpoint
		self.gg = tf.Graph()
		self.gg.as_default()
		self.sess = tf.Session()
		self.restored = False
		self.images_pl = tf.placeholder(tf.string, shape=())
		#self.inferi(np.array(Image.open('25.png')))

	def infer(self, path_to_image_file):
		with tf.Graph().as_default():
			image = Image.open(path_to_image_file)
			image = image.resize([64,64])
			image = tf.decode_raw(image.tobytes(), tf.uint8)
			image = tf.reshape(image, [64, 64, 3])
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)
			image = tf.image.resize_images(image, [54, 54])
			images = tf.reshape(image, [1, 54, 54, 3])

			length_logits, digits_logits = Model.inference(images)
			length_predictions = tf.argmax(length_logits, axis=1)
			digits_predictions = tf.argmax(digits_logits, axis=2)

			with tf.Session() as sess:
				sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

				restorer = tf.train.Saver()
				restorer.restore(sess, self.checkpoint)

				length_predictions_val, digits_predictions_string_val = sess.run([length_predictions, digits_predictions])
				length_prediction_val = length_predictions_val[0]
				digits_prediction_string_val = digits_predictions_string_val[0]
				#print(length_prediction_val)
				ans = []
				for i in range(int(length_predictions_val)):
					ans.append(int(digits_prediction_string_val[i]) % 10)
				#print(ans)

		return ans

	def inferi(self, img):
		image = Image.fromarray(img, 'RGB')
		image = image.resize([64,64])

		feed_dict = {
			self.images_pl: image.tobytes()
		}

		length_predictions_val, digits_predictions_string_val = self.sess.run([self.length_predictions, self.digits_predictions], feed_dict=feed_dict)
		length_prediction_val = length_predictions_val[0]
		digits_prediction_string_val = digits_predictions_string_val[0]
		#print(length_prediction_val)
		ans = []
		for i in range(int(length_predictions_val)):
			ans.append(int(digits_prediction_string_val[i]) % 10)
		#print(ans)

		return ans

	def init_graph(self):
		image = tf.decode_raw(self.images_pl, tf.uint8)
		image = tf.reshape(image, [64, 64, 3])
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize_images(image, [54, 54])
		images = tf.reshape(image, [1, 54, 54, 3])
		length_logits, digits_logits = Model.inference(images)
		self.length_predictions = tf.argmax(length_logits, axis=1)
		self.digits_predictions = tf.argmax(digits_logits, axis=2)
		if not self.restored:
			restorer = tf.train.Saver()
			self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
			restorer.restore(self.sess, self.checkpoint)
			self.restored = True
