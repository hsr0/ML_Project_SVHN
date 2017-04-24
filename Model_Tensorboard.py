import tensorflow as tf

class Model(object):
	@staticmethod
	def inference(x, y=0.2):
		#Layer 1
		with tf.name_scope('Hidden1') as scope:
			conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[5,5],padding='same',name='conv1')
			norm1 = tf.layers.batch_normalization(conv1,name='batch_norm1')
			actv1 = tf.nn.relu(norm1,name='relu1')
			pool1 = tf.layers.max_pooling2d(actv1, pool_size=[2,2],strides=2,padding='same',name='max_pool1')
			layer1 = tf.layers.dropout(pool1, rate=y,name='dropout1')

		#Layer 2
		with tf.name_scope('Hidden2') as scope:
			conv2 = tf.layers.conv2d(layer1, filters=64, kernel_size=[5,5],padding='same',name='conv2')
			norm2 = tf.layers.batch_normalization(conv2,name='batch_norm2')
			actv2 = tf.nn.relu(norm2,name='relu2')
			pool2 = tf.layers.max_pooling2d(actv2, pool_size=[2,2],strides=2,padding='same',name='max_pool2')
			layer2 = tf.layers.dropout(pool2, rate=y,name='dropout2')

		#Layer 3
		with tf.name_scope('Hidden3') as scope:
			conv3 = tf.layers.conv2d(layer2, filters=128, kernel_size=[5,5],padding='same',name='conv3')
			norm3 = tf.layers.batch_normalization(conv3,name='batch_norm3')
			actv3 = tf.nn.relu(norm3,name='relu3')
			pool3 = tf.layers.max_pooling2d(actv3, pool_size=[2,2],strides=1,padding='same',name='max_pool3')
			layer3 = tf.layers.dropout(pool3, rate=y,name='dropout3')

		#Layer 4
		with tf.name_scope('Hidden4') as scope:
			conv4 = tf.layers.conv2d(layer3, filters=128, kernel_size=[7,7],padding='same',name='conv4')
			norm4 = tf.layers.batch_normalization(conv4,name='batch_norm4')
			actv4 = tf.nn.relu(norm4,name='relu4')
			pool4 = tf.layers.max_pooling2d(actv4, pool_size=[2,2],strides=2,padding='same',name='max_pool4')
			layer4 = tf.layers.dropout(pool4, rate=y,name='dropout4')

		#Layer 5
		with tf.name_scope('Hidden5') as scope:
			conv5 = tf.layers.conv2d(layer4, filters=256, kernel_size=[7,7],padding='same',name='conv5')
			norm5 = tf.layers.batch_normalization(conv5,name='batch_norm5')
			actv5 = tf.nn.relu(norm5,name='relu5')
			pool5 = tf.layers.max_pooling2d(actv5, pool_size=[2,2],strides=1,padding='same',name='max_pool5')
			layer5 = tf.layers.dropout(pool5, rate=y,name='dropout5')

		#Layer 6
		with tf.name_scope('Hidden6') as scope:
			conv6 = tf.layers.conv2d(layer5, filters=256, kernel_size=[5,5],padding='same',name='conv6')
			norm6 = tf.layers.batch_normalization(conv6,name='batch_norm6')
			actv6 = tf.nn.relu(norm6,name='relu6')
			pool6 = tf.layers.max_pooling2d(actv6, pool_size=[2,2],strides=2,padding='same',name='max_pool6')
			layer6 = tf.layers.dropout(pool6, rate=y,name='dropout6')

		flatten = tf.reshape(layer6,[-1,4*4*256])

		#Fully Connected Layer
		with tf.name_scope('Hidden7') as scope:
			dense1 = tf.layers.dense(flatten, units=4096, activation=tf.nn.relu)

		with tf.name_scope('Hidden8') as scope:
			dense = tf.layers.dense(dense1, units=4096, activation=tf.nn.tanh)

		with tf.name_scope('Length') as scope:
			length = tf.layers.dense(dense, units=8)

		with tf.name_scope('digit1') as scope:
			digit1 = tf.layers.dense(dense, units=11)

		with tf.name_scope('digit2') as scope:
			digit2 = tf.layers.dense(dense, units=11)

		with tf.name_scope('digit3') as scope:
			digit3 = tf.layers.dense(dense, units=11)

		with tf.name_scope('digit4') as scope:
			digit4 = tf.layers.dense(dense, units=11)

		with tf.name_scope('digit5') as scope:
			digit5 = tf.layers.dense(dense, units=11)

		with tf.name_scope('digit6') as scope:
			digit6 = tf.layers.dense(dense, units=11)

		digits = tf.stack([digit1,digit2,digit3,digit4,digit5,digit6],axis=1)

		return length,digits

	@staticmethod
	def loss(length_predict,digits_predict,length_batch,digits_batch):
		length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_batch, logits=length_predict))
		digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,0],logits=digits_predict[:,0,:]))
		digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,1],logits=digits_predict[:,1,:]))
		digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,2],logits=digits_predict[:,2,:]))
		digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,3],logits=digits_predict[:,3,:]))
		digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,4],logits=digits_predict[:,4,:]))
		digit6_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_batch[:,5],logits=digits_predict[:,5,:]))
		return length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy + digit6_cross_entropy
