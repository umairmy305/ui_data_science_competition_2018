# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 02:17:05 2020

@author: umair

This file contains th code that will read the data for the UI 2018 data science
competition and output the results; 0 for wrong and 1 for correct.
The data consists of mathematical expressions with either a '+' or '-' operator
where the final answer can be in the last or first column.
E.g. 5 = 3 + 2 or 3 + 2 = 6, with the 1st expression correct and 2nd one wrong.
A convolutional neural network is used along with data augmentation 
"""
##################### Import important libraries ##############################

import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)

path = os.getcwd()

############################# Import Data #####################################
Train_data=pd.read_csv(path+'\\train.csv')
Train_labels=pd.read_csv(path+'\\train_labels.csv')
Test_data=pd.read_csv(path+'\\test.csv')

############################# Pre-Processing ##################################
# Data re-organization and visualization
numOfPixels = 576
namesList = [[] for i in range(numOfPixels)]
namesList = ['pixel '+str(i+1) for i in range(numOfPixels)]
x_train = Train_data[namesList].values
x_trshp = np.reshape(x_train,[80000,24,24])
X0 = np.copy(x_trshp)
X1 = np.roll(X0,-1,axis=2)
X1[:,:,-1] = np.zeros([24])
X2 = np.roll(X0,1,axis=2)
X2[:,:,0] = np.zeros([24])
X3 = np.roll(X0,-1,axis=1)
X3[:,-1,:] = np.zeros([24])
X4 = np.roll(X0,1,axis=1)
X4[:,0,:] = np.zeros([24])
y_train_col = np.uint8(Train_labels[['label']].values)
y_train_col = y_train_col[:,0]
y_train = np.zeros([80000,13])
y_train[np.arange(80000), y_train_col] = 1
print("The training image and label shapes:")
print(x_trshp.shape)
print(y_train.shape)
print("Training Image Examples followed by one-hot labels:")
for i in range(1,7):
    plt.subplot(1,6,i)
    plt.imshow(1-X0[i], cmap = plt.cm.gray, interpolation='nearest')
plt.show()
print(y_train[0:5])


namesListTestAll = ['pixel '+str(i+1) for i in range(5*numOfPixels)]
namesListTest1 = [namesListTestAll[120*i+j] for i in range(24) for j in range(24)]
namesListTest2 = [namesListTestAll[(120*i+24)+j] for i in range(24) for j in range(24)]
namesListTest3 = [namesListTestAll[(120*i+48)+j] for i in range(24) for j in range(24)]
namesListTest4 = [namesListTestAll[(120*i+72)+j] for i in range(24) for j in range(24)]
namesListTest5 = [namesListTestAll[(120*i+96)+j] for i in range(24) for j in range(24)]

x_testAll = Test_data[namesListTestAll].values
x_test1 = Test_data[namesListTest1].values
x_test2 = Test_data[namesListTest2].values
x_test3 = Test_data[namesListTest3].values
x_test4 = Test_data[namesListTest4].values
x_test5 = Test_data[namesListTest5].values
print("The test data shapes:")
print(x_testAll.shape)
print("Test Image Examples:")
for i in range(1,7):
    plt.subplot(6,1,i)
    plt.imshow(1-np.reshape(x_testAll[i],[24, 120]), cmap = plt.cm.gray, interpolation='nearest')
plt.show()

#################### Feature Extraction/Selection #############################
#### Data Augmentation to increase the number of samples
CC = 64000; # 80% of training images (80000) 
x_traugT = np.array([X0[:CC],X1[:CC],X2[:CC],X3[:CC],X4[:CC]])
x_traugT = x_traugT.reshape(5*CC,24,24)
x_traugV = np.array([X0[CC:],X1[CC:],X2[CC:],X3[CC:],X4[CC:]])
x_traugV = x_traugV.reshape(5*(80000-CC),24,24)
x_traug = np.append(x_traugT, x_traugV, axis = 0)
print('Augmented data shape, original data shape, and final data shape.')
print(x_traugT.shape, x_traugV.shape, x_traug.shape)
print('Example of the first two images augmented in 4 directions:')
for i in range(1,6):
    plt.subplot(2,5,i)
    plt.imshow(1-x_traug[(i-1)*CC+1], cmap = plt.cm.gray, interpolation='nearest')
    #plt.show()
    plt.subplot(2,5,i+5)
    plt.imshow(1-x_traug[5*CC+(i-1)*(80000-CC)+2], cmap = plt.cm.gray, interpolation='nearest')
plt.show()
    
y_traugT = np.array([y_train[:CC],y_train[:CC],y_train[:CC],y_train[:CC],y_train[:CC]])
y_traugT = y_traugT.reshape(5*CC,13)
y_traugV = np.array([y_train[CC:],y_train[CC:],y_train[CC:],y_train[CC:],y_train[CC:]])
y_traugV = y_traugV.reshape(5*(80000-CC),13)
y_traug = np.append(y_traugT, y_traugV, axis = 0)
print('Augmented label shape, original label shape, and final labels:')
print(y_traugT.shape, y_traugV.shape, y_traug.shape)
print('One-hot labels of the first two augmented images:')
for i in range(5):
    print(y_traug[i*CC:i*CC+2])

# Split data for training and validation
N_train = 320000
X_valid = x_traug[N_train:]
X_train = x_traug[0:N_train]
Y_valid = y_traug[N_train:]
Y_train = y_traug[0:N_train]

############################# Classification ##################################

# Define the number of convolutional /fully-connected (FC) neural network layers
n0 = 576
n1 = 36
n2 = 60
n3 = 72
n4 = 124
n5 = 36
n6 = 13

# Create a TensorFlow interactive session to update learning rate
sess = tf.InteractiveSession()

# Create input, output, weights and biases vectors

X_ = tf.placeholder(tf.float32, shape = [None, 24, 24, 1])
Y_ = tf.placeholder(tf.float64, shape = [None, n6])

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, n1], stddev = 1/np.sqrt(n0)))
W2 = tf.Variable(tf.truncated_normal([3, 3, n1, n2], stddev = 1/np.sqrt(n1)))
W3 = tf.Variable(tf.truncated_normal([4, 4, n2, n3], stddev = 1/np.sqrt(n2)))
W4 = tf.Variable(tf.truncated_normal([6*6*n2, n4], stddev = 1/np.sqrt(n3)))
W5 = tf.Variable(tf.truncated_normal([n4, n5], stddev = 1/np.sqrt(n4)))
W6 = tf.Variable(tf.truncated_normal([n5, n6], stddev = 1/np.sqrt(n5)))
b1 = tf.Variable([tf.constant(0.0, shape=[n1])])
b2 = tf.Variable([tf.constant(0.0, shape=[n2])])
b3 = tf.Variable([tf.constant(0.0, shape=[n3])])
b4 = tf.Variable([tf.constant(0.0, shape=[n4])])
b5 = tf.Variable([tf.constant(0.0, shape=[n5])])
b6 = tf.Variable([tf.constant(0.0, shape=[n6])])

##### First convolutional layer with max-pooling and normalization
stride = 1  # output is 12x12
Y1 = tf.nn.relu(tf.nn.conv2d(X_, W1, strides=[1, stride, stride, 1], padding='SAME')+b1)
mean1, variance1 = tf.nn.moments(Y1, axes = [0])
Offset1 = tf.Variable(tf.zeros([1, n1]))
Scales1 = tf.Variable(tf.ones([1, n1]))
Y1nor = tf.nn.batch_normalization(Y1, mean1, variance1, offset=Offset1, scale = Scales1, variance_epsilon = 1e-16)
Y1max = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');

##### Second convolutional layer with max-pooling and normalization
stride = 1  # output is 6x6
Y2 = tf.nn.relu(tf.nn.conv2d(Y1max, W2, strides=[1, stride, stride, 1], padding='SAME')+b2)
mean2, variance2 = tf.nn.moments(Y2, axes = [0])
Offset2 = tf.Variable(tf.zeros([1, n2]))
Scales2 = tf.Variable(tf.ones([1, n2]))
Y2nor = tf.nn.batch_normalization(Y2, mean2, variance2, offset=Offset2, scale = Scales2, variance_epsilon = 1e-16)
Y2max = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');

##### Third convolutional layer
stride = 2  # output is 3x3
Y3 = tf.nn.relu(tf.nn.conv2d(Y2max, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)

# Reshape for input to FC
YY = tf.reshape(Y2max, shape=[-1, 6 * 6 * n2])

##### First FC layer
Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
mean4, variance4 = tf.nn.moments(Y4, axes = [0])
Offset4 = tf.Variable(tf.zeros([1, n4]))
Scales4 = tf.Variable(tf.ones([1, n4]))
Y4nor = tf.nn.batch_normalization(Y4, mean4, variance4, offset=Offset4, scale = Scales4, variance_epsilon = 1e-16)

##### Second FC layer
Y5 = tf.nn.relu(tf.matmul(Y4nor, W5) + b5)
mean5, variance5 = tf.nn.moments(Y5, axes = [0])
Offset5 = tf.Variable(tf.zeros([1, n5]))
Scales5 = tf.Variable(tf.ones([1, n5]))
Y5nor = tf.nn.batch_normalization(Y5, mean5, variance5, offset=Offset5, scale = Scales5, variance_epsilon = 1e-16)

##### Outputs
Ylogits = tf.matmul(Y5nor, W6) + b6
Y_hat = tf.cast(tf.nn.softmax(Ylogits), tf.float64)

##### Define the CNN hyper-parameters
init = tf.global_variables_initializer()
losses_matrix = tf.cast(Y_ * tf.log(Y_hat+1e-16), tf.float64)
cross_entropy = tf.cast(-tf.reduce_sum(losses_matrix), tf.float64)
is_correct = tf.equal(tf.argmax(Y_hat,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

learning_rate_ = tf.placeholder(tf.float32, shape =[])
learning_rate_decay = 0.9
optimizer3 = tf.train.MomentumOptimizer(learning_rate=learning_rate_,momentum = 0.9)

train_step = optimizer3.minimize(cross_entropy)
sess = tf.Session()
sess.run(init)
sess.run(tf.global_variables_initializer())

batch_size = 50
batch_size_test = 200;
test_runs = int((400000-N_train)/batch_size_test)
train_size = X_train.shape[0]
itersPerEpochs = int(train_size/batch_size)
epochs = 5;
eta = 0.001
#iters_per_epoch 
for j in range(epochs+1):
    perm=np.arange(train_size)
    np.random.shuffle(perm)
    for i in range(itersPerEpochs):
        k = i%itersPerEpochs
        batch_X = np.reshape(X_train[k*batch_size:(k+1)*batch_size], [batch_size, 24, 24, 1])
        batch_Y = Y_train[k*batch_size:(k+1)*batch_size]
        train_data = {X_: batch_X, Y_: batch_Y, learning_rate_:eta}
        #Z = sess.run(Y3, feed_dict = train_data)
        #print(Z.shape)
        sess.run(train_step, feed_dict = train_data)
        a, c = sess.run([accuracy, cross_entropy], feed_dict = train_data)
    #print('Epoch#: %03d, Loss: %7.6f, Accuracy: %4.0f%%' % (j, c, a*100)) 
    sys.stdout.write('Epoch#: %03d, Loss: %12.10f, Accuracy: %4.0f%%\n' % (j, c, a*100))
    eta*=learning_rate_decay
    if (j%5 == 0):
        accCount = np.zeros([400000-N_train])
        for i in range(test_runs):
            test_data = {X_:np.reshape(X_valid[i*200:(i+1)*200],[200,24,24,1]), Y_:Y_valid[i*200:(i+1)*200]}
            accCount[i*200:(i+1)*200] = sess.run(is_correct, feed_dict = test_data)
        #at = sess.run(accuracy, feed_dict = np.reshape(test_data))
        at = np.sum(1*accCount)/(400000-N_train)
        sys.stdout.write('Epoch#: %03d, Validation Accuracy: %0.3f%%\n' % (j, at*100))

############################### Evaluation ####################################
Q1 = np.zeros([20000,13])
Q2 = np.zeros([20000,13])
Q3 = np.zeros([20000,13])
Q4 = np.zeros([20000,13])
Q5 = np.zeros([20000,13])
for i in range(100):
    test_data1 = {X_:np.reshape(x_test1[i*200:(i+1)*200],[200,24,24,1])}
    test_data2 = {X_:np.reshape(x_test2[i*200:(i+1)*200],[200,24,24,1])}
    test_data3 = {X_:np.reshape(x_test3[i*200:(i+1)*200],[200,24,24,1])}
    test_data4 = {X_:np.reshape(x_test4[i*200:(i+1)*200],[200,24,24,1])}
    test_data5 = {X_:np.reshape(x_test5[i*200:(i+1)*200],[200,24,24,1])}
    Q1[i*200:(i+1)*200] = sess.run(Y_hat, feed_dict = test_data1)
    Q2[i*200:(i+1)*200] = sess.run(Y_hat, feed_dict = test_data2)
    Q3[i*200:(i+1)*200] = sess.run(Y_hat, feed_dict = test_data3)
    Q4[i*200:(i+1)*200] = sess.run(Y_hat, feed_dict = test_data4)
    Q5[i*200:(i+1)*200] = sess.run(Y_hat, feed_dict = test_data5)
    
y_pred1 = np.argmax(Q1[:,0:10], axis = 1)
y_pred2 = np.argmax(Q2[:,10:], axis = 1)+10
y_pred3 = np.argmax(Q3[:,0:10], axis = 1)
y_pred4 = np.argmax(Q4[:,10:], axis = 1)+10
y_pred5 = np.argmax(Q5[:,0:10], axis = 1)
finalLabels = np.zeros_like(y_pred1)

for i in range(20000):
    if (y_pred2[i] == 12):
        if (y_pred4[i] == 10):
            if ((y_pred3[i] + y_pred5[i])==y_pred1[i]):
                finalLabels[i] = 1
            else:
                finalLabels[i] = 0
        else:
            if ((y_pred3[i] - y_pred5[i])==y_pred1[i]):
                finalLabels[i] = 1
            else:
                finalLabels[i] = 0
    else:
        if (y_pred2[i] == 10):
            if ((y_pred1[i] + y_pred3[i])==y_pred5[i]):
                finalLabels[i] = 1
            else:
                finalLabels[i] = 0
        else:
            if ((y_pred1[i] - y_pred3[i])==y_pred5[i]):
                finalLabels[i] = 1
            else:
                finalLabels[i] = 0
indexOuts = np.arange(20000)
arrayOuts = np.transpose(np.array([indexOuts, finalLabels]))
arrayOuts.shape
############################### Submission ####################################
# the final accuracy was calculated by the online submission system of the 
# competition. We just submitted a 'csv' file.
np.savetxt('umairSubmission16.csv', arrayOuts, fmt='%d', delimiter=',', header="index,label")