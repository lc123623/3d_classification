from __future__ import division
import tensorflow as tf 
import numpy as np 


BATCH_SIZE=128
EPOCHS=100
SAMPLE_SIZE=60000



def variable_on_cpu(name,shape):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
	return var

def variable_with_decay(name,shape,decay_weight):
	var=variable_on_cpu(name,shape)
	if decay_weight!=0:
		decay=tf.multiply(tf.nn.l2_loss(var),decay_weight,name='weight_loss')
		tf.add_to_collection('loss',decay)
	return var


def inference(input):
	with tf.variable_scope('conv1') as scope:
		kernel=variable_with_decay('weights',[4,4,4,1,64],decay_weight=0)
		conv=tf.nn.conv3d(input,kernel,strides=[1,2,2,2,1],padding='SAME')
		biases=variable_with_decay('biases',[64],decay_weight=0)
		conv1=tf.add(conv,biases)
		norm=tf.layers.batch_normalization(conv1)
		conv1=tf.nn.relu(norm)
	with tf.variable_scope('conv2') as scope:
		kernel=variable_with_decay('weights',[4,4,4,64,128],decay_weight=0)
		conv=tf.nn.conv3d(conv1,kernel,strides=[1,2,2,2,1],padding='SAME')
		biases=variable_with_decay('biases',[128],decay_weight=0)
		conv2=tf.add(conv,biases)
		norm=tf.layers.batch_normalization(conv2)
		conv2=tf.nn.relu(norm)
	with tf.variable_scope('conv3') as scope:
		kernel=variable_with_decay('weights',[4,4,4,128,256],decay_weight=0)
		conv=tf.nn.conv3d(conv2,kernel,strides=[1,2,2,2,1],padding='SAME')
		biases=variable_with_decay('biases',[256],decay_weight=0)
		conv3=tf.add(conv,biases)
		norm=tf.layers.batch_normalization(conv3)
		conv3=tf.nn.relu(norm)
	reshape=tf.reshape(conv3,[BATCH_SIZE,16384])
	with tf.variable_scope('fc1') as scope:
		weight=variable_with_decay('weights',[16384,5000],decay_weight=0.001)
		fc=tf.matmul(reshape,weight)
		biases=variable_with_decay('biases',[5000],decay_weight=0)
		fc1=tf.add(fc,biases)
		fc1=tf.nn.dropout(fc1,keep_prob=keep_prob)
	with tf.variable_scope('fc2') as scope:
		weight=variable_with_decay('weights',[5000,500],decay_weight=0.001)
		fc=tf.matmul(fc1,weight)
		biases=variable_with_decay('biases',[500],decay_weight=0)
		fc2=tf.add(fc,biases)
		fc2=tf.nn.dropout(fc2,keep_prob=keep_prob)
	with tf.variable_scope('fc3') as scope:
		weight=variable_with_decay('weights',[500,10],decay_weight=0.001)
		fc=tf.matmul(fc2,weight)
		biases=variable_with_decay('biases',[10],decay_weight=0)
		fc3=tf.add(fc,biases)
	tf.get_variable_scope().reuse_variables()
	return fc3

def read_tfrecords(tfrecords_file):
	filename_quene=tf.train.string_input_producer([tfrecords_file])
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_quene)
	features=tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([1], tf.int64),
								      'data' : tf.FixedLenFeature([27000], tf.int64)
								      })
	train_data=tf.reshape(features['data'],(30,30,30,1))
	train_data=tf.cast(train_data,tf.float32)
	label_data=tf.reshape(tf.one_hot(indices=features['label'],depth=10,axis=-1),[10])
	label_data=tf.cast(label_data,tf.float32)
	return train_data , label_data


def compute_accuracy(output,label):
	 output=np.argmax(output,axis=1)
	 label=np.argmax(label,axis=1)
	 count=0
	 for ii in range(BATCH_SIZE):
	 	if output[ii]==label[ii]:
	 		count+=1
	 return count/BATCH_SIZE



def generate_network(train=True):
	with tf.device('/cpu:0'):
		train_input,train_label=read_tfrecords('train.tfrecords')
		train_input,train_label=tf.train.shuffle_batch([train_input,train_label],batch_size=BATCH_SIZE,capacity=40000,min_after_dequeue=10000)
		test_input,test_label=read_tfrecords('test.tfrecords')
		test_input,test_label=tf.train.shuffle_batch([test_input,test_label],batch_size=BATCH_SIZE,capacity=10000,min_after_dequeue=5000)
		global_step = tf.Variable(0)
		keep_prob=tf.placeholder(tf.float32)
		
	with tf.device('/gpu:0'):
		train_logits=inference(train_input)
		train_out=tf.nn.softmax(train_logits)
		#print 'train_label',train_label
		train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label,logits=train_logits))
		tf.add_to_collection('loss',train_loss)
		cost=tf.add_n(tf.get_collection('loss'))
		
		test_logits=inference(test_input)
		test_output=tf.nn.softmax(test_logits)
		test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_label,logits=test_logits))
		update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#for ii in tf.trainable_variables():
		#	print ii.name
		        
		  
  		learning_rate = tf.train.exponential_decay(5e-4,global_step,decay_steps=SAMPLE_SIZE/BATCH_SIZE,decay_rate=0.98,staircase=True)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)
		
		#test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_label,logits=test_logits))
	tf.summary.scalar('training cost',cost)
	tf.summary.scalar('learning rate',learning_rate)
	merged=tf.summary.merge_all()
	saver=tf.train.Saver()
	sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
	threads=tf.train.start_queue_runners(sess=sess)
	writer=tf.summary.FileWriter('./log/log',sess.graph)
	if train==False:
		model_file=tf.train.latest_checkpoint('./model/')
		saver.restore(sess,model_file)
	elif train=True:
		sess.run(tf.global_variables_initializer())
	#np.set_printoptions(precision=4,threshold=np.NaN)
	return  train_loss,cost,optimizer,merged,global_step,train_label,train_out
	
	'''
	for ii in range(400):
		train_input_,train_label_,test_input_,test_label_=sess.run([train_input,train_label,test_input,test_label])
	'''
	

		

