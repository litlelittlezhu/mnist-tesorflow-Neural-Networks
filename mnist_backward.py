#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200 #定义批处理数
LEARNING_RATE_BASE = 0.1 #定义基础学习速率
LEARNING_RATE_DECAY = 0.99 #定义学习速率下降率
REGULARIZER = 0.0001 #定义正则化系数
STEPS = 50000 #定义训练次数
MOVING_AVERAGE_DECAY = 0.99 #定义滑动平均系数
MODEL_SAVE_PATH = '/home/zbf/mnist/model'#定义模型保存路径
MODEL_NAME = 'mnist_model'#定义模型文件名字

def backward(mnist):
	x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
	y_= tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	loss = cem+tf.add_n(tf.get_collection('losses'))#损失函数
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)#衰减型学习速率
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)#梯度下降法权重更新

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)#定义滑动平均
	#tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True/False)
	ema_op = ema.apply(tf.trainable_variables())#对所有变量实行滑动平均

        #确保train_step,ema_op按顺序都执行
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name='train')
	
	saver = tf.train.Saver()#创建saver
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:#断点续训功能
			saver.restore(sess,ckpt.model_checkpoint_path)
		
		for i in range(STEPS):
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			_,loss_value ,step= sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			if i%1000==0:
				print('After %d trining step(s),loss on trining batch is %g'%(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)#保存训练模型

def main():
	mnist = input_data.read_data_sets('/home/zbf/mnist',one_hot=True)
	backward(mnist)
if __name__ == '__main__':
	main()

