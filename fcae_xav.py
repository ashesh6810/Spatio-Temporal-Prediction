
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

train=genfromtxt('train_input1.csv',delimiter=",");
X_train=np.transpose(train)

test=genfromtxt('train_label1.csv',delimiter=",");
X_test=np.transpose(test)

# Training Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 32
dropout = 0.5
display_step = 10
examples_to_show = 10
keep_prob = tf.placeholder(tf.float32)

# Network Parameters
num_hidden_1 = 100 # 1st layer num features
num_hidden_2 = 50 # 1st layer num features
num_hidden_3 = 10 # 2nd layer num features (the latent dim)
num_input = 41 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_input])

initializer = tf.contrib.layers.xavier_initializer()
weights = {
    'encoder_h1': tf.Variable(initializer([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(initializer([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(initializer([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(initializer([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(initializer([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(initializer([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(initializer([num_hidden_1])),
    'encoder_b2': tf.Variable(initializer([num_hidden_2])),
    'encoder_b3': tf.Variable(initializer([num_hidden_3])),
    'decoder_b1': tf.Variable(initializer([num_hidden_2])),
    'decoder_b2': tf.Variable(initializer([num_hidden_1])),
    'decoder_b3': tf.Variable(initializer([num_input])),
}

# Building the encoder
def encoder(x,dropout):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_3=tf.nn.dropout(layer_3,dropout)
    return layer_3


# Building the decoder
def decoder(x,dropout):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_3=tf.nn.dropout(layer_3,dropout)
    return layer_3

# Construct model
encoder_op = encoder(X,keep_prob)
decoder_op = decoder(encoder_op,keep_prob)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
#initializer = tf.global_variables_init()
initializer = tf.contrib.layers.xavier_initializer() 
# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(initializer)

    # Training
    for i in range(1, num_steps+1):
      print(i)  
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
      for step2 in range(0,int(np.size(X_train,0)/batch_size)):  
       print(step2)
       batch_x=(X_train[range(step2*batch_size,(step2+1)*batch_size),:])
       batch_y=(X_test[range(step2*batch_size,(step2+1)*batch_size),:])
        # Run optimization op (backprop) and cost op (to get loss value)
       _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,Y:batch_y,keep_prob: dropout})
        # Display logs per step
  #     if i % display_step == 0 or i == 1:
       print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    input=genfromtxt('test_input1.csv',delimiter=",")  
    label=genfromtxt('test_labels1.csv',delimiter=",")  
    
#    inputT=(X_train[1:10,:])
    inputT=np.transpose(input[:,0:10])
    g=sess.run(decoder_op,feed_dict={X:inputT,keep_prob: 1.0})
    print(np.shape(g))

    for j in range(0,10):
     plt.figure
     plt.plot(g[j,:],color='red')
     plt.plot(label[:,j],color='black')
     plt.savefig('time_step'+ str(j)+'.png')
     plt.plot()
