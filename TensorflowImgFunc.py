
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import os

fname = 'luk.jpg'


XOR_X = np.empty([10000,2])
XOR_Y = np.empty([10000,3])
im = Image.open(fname)
im.thumbnail((100,100), Image.ANTIALIAS)
I = np.asarray(im)
I.flags.writeable=True

for x in range(100):
    for y in range(100):
        XOR_Y[(y+(x*100)),0] = ((I[x,y,0]-128)/256)*2 
        XOR_Y[(y+(x*100)),1] = ((I[x,y,1]-128)/256)*2 
        XOR_Y[(y+(x*100)),2] = ((I[x,y,2]-128)/256)*2 

        XOR_X[(y+(x*100)),0] = (x-50)/50
        XOR_X[(y+(x*100)),1] = (y-50)/50
  

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[10000,2])
    y = tf.placeholder(tf.float32,shape=[10000,3])

with tf.name_scope('weights'):
    w1 = tf.Variable(tf.random_normal([2, 15]), name="W1")
    w2 = tf.Variable(tf.random_normal([15, 15]), name="W2")
    w3 = tf.Variable(tf.random_normal([15, 15]), name="W3")
    w4 = tf.Variable(tf.random_normal([15, 3]), name="W4")

with tf.name_scope('biases'):
    b1 = tf.Variable(tf.random_normal([1, 15]), name="b1")
    b2 = tf.Variable(tf.random_normal([1, 15]), name="b2")
    b3 = tf.Variable(tf.random_normal([1, 15]), name="b3")
    b4 = tf.Variable(tf.random_normal([1, 3]), name="b4")

with tf.name_scope('l1'):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, w1), b1))

with tf.name_scope('l2'):
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))

with tf.name_scope('l3'):
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3), b3))

with tf.name_scope('l4'):
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, w4), b4))

with tf.name_scope('regularization'):
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)+ tf.nn.l2_loss(w4)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(layer_4 - y)) + 0.0000001 * regularization
    loss = tf.Print(loss, [loss], "loss")

with tf.name_scope('trainer'):
    train_op = tf.train.GradientDescentOptimizer(0.7).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init)


    for i in range(1000000009):
        session.run(train_op, feed_dict={x: XOR_X, y: XOR_Y})

        if i % 1000 == 0:
            I = np.empty([100,100,3])
            res = session.run(layer_4, feed_dict={x: XOR_X, y: XOR_Y})
            for x_ in range(100):
                for y_ in range(100):
                    I[x_,y_,0] = (((res[(y_+(x_*100)),0] + 1) / 2) * 255) 
                    I[x_,y_,1] = (((res[(y_+(x_*100)),1] + 1) / 2) * 255)  
                    I[x_,y_,2] = (((res[(y_+(x_*100)),2] + 1) / 2) * 255) 
            ima = Image.fromarray(np.uint8(I))
            plt.axis("off")
            plt.imshow(ima)    
            plt.draw()
            plt.pause(0.01)
