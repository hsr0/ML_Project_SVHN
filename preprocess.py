import tensorflow as tf
import os
import csv
import numpy as np
from PIL import Image
import random

class PreProcess(object):
	def __init__ (self, file_dir):
		# Use TF to load image data from files
		self._path_to_images = os.path.join(file_dir, 'train_images/')
		self._cropped = os.path.join(file_dir, 'cropped_images/')
		self._csv_path = os.path.join(file_dir, 'train.csv')
		self._train = os.path.join(file_dir, 'train.tfrecords')
		self._val = os.path.join(file_dir, 'val.tfrecords')
		self._list = []
		path_to_image_files = tf.gfile.Glob(os.path.join(file_dir, 'train_images/*.png'))
		for _ in range(len(path_to_image_files)):
			info = {
				"fname" : "",
				"num": 0,
				"label": [10,10,10,10,10,10],
				"hmin": 0,
				"wmin":	0,
				"hmax": 0,
				"wmax": 0,
			}
			self._list.append(info)


	def _trim(self, img, x1, y1, x2, y2):
		w = x2 - x1
		h = y2 - y1
		xx1 = round(x1 - 0.15 * w)
		yy1 = round(y1 - 0.15 * h)
		xx2 = round(x2 + 0.15 * w)
		yy2 = round(y2 + 0.15 * h)
		img = img.crop([xx1,yy1,xx2,yy2])
		img = img.resize([64,64])
		return img

	def write_back(self):
		n1 = 0
		writer = tf.python_io.TFRecordWriter(self._train)
		writer1 = tf.python_io.TFRecordWriter(self._val)
		for li in self._list:
			img = self._trim(Image.open(os.path.join(self._path_to_images, li["fname"])),li["wmin"],li["hmin"],li["wmax"],li["hmax"])
			image = img.tobytes()
			example = tf.train.Example(features=tf.train.Features(feature={
				'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
				'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[li["num"]])),
				'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=li["label"]))
				}))
			if(random.random() > 0.05):
				writer.write(example.SerializeToString())
			else:
				writer1.write(example.SerializeToString())
				n1 += 1
			#img.save(os.path.join(self._cropped, li["fname"]))
		print("Num of val records : " + str(n1))
		writer.close()
		writer1.close()

	def load_CSV(self):
		with open(self._csv_path, newline='') as csvfile:
			digit_reader = csv.reader(csvfile)
			first = 0
			for row in digit_reader:
				if first == 0:
					first = 1
					continue
				index = int(row[0].split('.')[0]) - 1;
				num = 0 if int(row[1]) == 10 else int(row[1])
				if(self._list[index]["num"] > 5):
					continue
				self._list[index]["label"][self._list[index]["num"]] = num;
				self._list[index]["num"] += 1
				self._list[index]["fname"] = row[0]
				if(self._list[index]["num"] == 1):
					self._list[index]["hmin"] = int(row[3])
					self._list[index]["wmin"] = int(row[2])
					self._list[index]["hmax"] = int(row[5]) + int(row[3])
					self._list[index]["wmax"] = int(row[4]) + int(row[2])
				else:
					self._list[index]["hmin"] = int(row[3] if self._list[index]["hmin"] > int(row[3]) else self._list[index]["hmin"])
					self._list[index]["wmin"] = int(row[2] if self._list[index]["wmin"] > int(row[2]) else self._list[index]["wmin"])
					self._list[index]["hmax"] = int(int(row[5]) + int(row[3]) if self._list[index]["hmax"] < int(row[5]) + int(row[3]) else self._list[index]["hmax"])
					self._list[index]["wmax"] = int(int(row[4]) + int(row[2]) if self._list[index]["wmax"] < int(row[4]) + int(row[2]) else self._list[index]["wmax"])


	def prepare_Data(self):
		self.load_CSV()
		self.write_back()

	def _retreive(self, data_queue):
		reader = tf.TFRecordReader()
		index, single_example = reader.read(data_queue)
		example = tf.parse_single_example(single_example, features={
			'image': tf.FixedLenFeature([], tf.string),
			'length': tf.FixedLenFeature([], tf.int64),
			'digits': tf.FixedLenFeature([6], tf.int64)
		})
		image = tf.decode_raw(example['image'], tf.uint8)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.reshape(image, [64, 64, 3])
		image = tf.random_crop(image, [54, 54, 3])
		return image, tf.cast(example['length'], tf.int32), tf.cast(example['digits'], tf.int32)

	def return_Data(self, batch_sz = 32):
		data_queue = tf.train.string_input_producer([self._train], num_epochs=None)
		image, length, digits = self._retreive(data_queue)

		return tf.train.shuffle_batch([image, length, digits], batch_size=batch_sz, num_threads=4, min_after_dequeue=100, capacity=1000, allow_smaller_final_batch=True)

	def return_val(self, batch_sz = 32):
		data_queue = tf.train.string_input_producer([self._val], num_epochs=None)
		image, length, digits = self._retreive(data_queue)

		return tf.train.batch([image, length, digits], batch_size=batch_sz, num_threads=4, capacity=1000, allow_smaller_final_batch=True)
