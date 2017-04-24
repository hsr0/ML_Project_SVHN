import tensorflow as tf

class Model(object):
	@staticmethod
	def inference(x, y=0.2):
		#Layer 1
		conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[5,5],padding='same')
		norm1 = tf.layers.batch_normalization(conv1)
		actv1 = tf.nn.relu(norm1)
		pool1 = tf.layers.max_pooling2d(actv1, pool_size=[2,2],strides=2,padding='same')
		layer1 = tf.layers.dropout(pool1, rate=y)

		#Layer 2
		conv2 = tf.layers.conv2d(layer1, filters=64, kernel_size=[5,5],padding='same')
		norm2 = tf.layers.batch_normalization(conv2)
		actv2 = tf.nn.relu(norm2)
		pool2 = tf.layers.max_pooling2d(actv2, pool_size=[2,2],strides=2,padding='same')
		layer2 = tf.layers.dropout(pool2, rate=y)

		#Layer 3
		conv3 = tf.layers.conv2d(layer2, filters=128, kernel_size=[5,5],padding='same')
		norm3 = tf.layers.batch_normalization(conv3)
		actv3 = tf.nn.relu(norm3)
		pool3 = tf.layers.max_pooling2d(actv3, pool_size=[2,2],strides=1,padding='same')
		layer3 = tf.layers.dropout(pool3, rate=y)

		#Layer 4
		conv4 = tf.layers.conv2d(layer3, filters=128, kernel_size=[7,7],padding='same')
		norm4 = tf.layers.batch_normalization(conv4)
		actv4 = tf.nn.relu(norm4)
		pool4 = tf.layers.max_pooling2d(actv4, pool_size=[2,2],strides=2,padding='same')
		layer4 = tf.layers.dropout(pool4, rate=y)

		#Layer 5
		conv5 = tf.layers.conv2d(layer4, filters=256, kernel_size=[7,7],padding='same')
		norm5 = tf.layers.batch_normalization(conv5)
		actv5 = tf.nn.relu(norm5)
		pool5 = tf.layers.max_pooling2d(actv5, pool_size=[2,2],strides=1,padding='same')
		layer5 = tf.layers.dropout(pool5, rate=y)

		#Layer 6
		conv6 = tf.layers.conv2d(layer5, filters=256, kernel_size=[5,5],padding='same')
		norm6 = tf.layers.batch_normalization(conv6)
		actv6 = tf.nn.relu(norm6)
		pool6 = tf.layers.max_pooling2d(actv6, pool_size=[2,2],strides=2,padding='same')
		layer6 = tf.layers.dropout(pool6, rate=y)

		flatten = tf.reshape(layer6,[-1,4*4*256])

		#Fully Connected Layer
		dense1 = tf.layers.dense(flatten, units=4096, activation=tf.nn.relu)
		dense = tf.layers.dense(dense1, units=4096, activation=tf.nn.tanh)

		length = tf.layers.dense(dense, units=8)
		digit1 = tf.layers.dense(dense, units=11)
		digit2 = tf.layers.dense(dense, units=11)
		digit3 = tf.layers.dense(dense, units=11)
		digit4 = tf.layers.dense(dense, units=11)
		digit5 = tf.layers.dense(dense, units=11)
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
