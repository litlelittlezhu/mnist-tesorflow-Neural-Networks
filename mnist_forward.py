#coding=utf-8
import tensorflow as tf
INPUT_NODE = 784 #输入节点为28*28=784	
OUTPUT_NODE = 10 #输出节点为10 对应0-9
LAYER1_NODE = 500 #只有一个隐含层 500个节点

def get_weight(shape,regularizer):#定义权重生成函数
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))#定义权重变量
	if regularizer != None:#该权重是否正则化
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	# 输出为(|1|+|-2|+|-3|+|4|)*0.5=5
        #print(sess.run(contrib.layers.l1_regularizer(0.5)(weight)))
        # 输出为(1²+(-2)²+(-3)²+4²)/2*0.5=7.5
        # TensorFlow会将L2的正则化损失值除以2使得求导得到的结果更加简洁
        #print(sess.run(contrib.layers.l2_regularizer(0.5)(weight)))
	return w

def get_bias(shape):#定义偏置
	b = tf.Variable(tf.zeros(shape))
	return b

def forward(x,regularizer):#定义前向传播过程
	w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1)+b1)#激活函数用relu
	
	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w2)+b2
	return y



