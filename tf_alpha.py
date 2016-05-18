import numpy as np
import glob
from PIL import Image
import cv2
from settings import *
from scripts import *
import tensorflow as tf

# select the image with 320 * 240 size
imginfo = []
with open(list_file) as f:
    for line in f:
        temp = line.split()
        if temp[1] == '320' and temp[2] == '240':
            imginfo.append(temp[0:-1])

# strat from small part of the data
nbsel = 100
imginfost = imginfo[0:nbsel]

# read the image and preprocess into 4-dimension matrix (batch, height, width, channel)
imgA = np.zeros((nbsel, 240, 320, 3))
imgB = np.zeros((nbsel, 120, 160, 3))
imgC = np.zeros((nbsel, 60, 80, 3))
for i in xrange(len(imginfost)):
    img = open_convert(image_dir + imginfost[i][0] + '.jpg', mode='YCbCr')
    lplist = laplacian_pyramid(img, 3)
    imgA[i] = lplist[0]
    imgB[i] = lplist[1]
    imgC[i] = lplist[2]
    
# local contrast normalisation
inputA = lecun_lcn_batch(imgA, kernel_shape=15, threshold=1e-4)
inputB = lecun_lcn_batch(imgB, kernel_shape=15, threshold=1e-4)
inputC = lecun_lcn_batch(imgC, kernel_shape=15, threshold=1e-4)

# get the label for every pixel of the image
img_label = read_label_batch(label_dir, imginfost, labeltype='regions')

# translate label into vector
label_in = np.zeros([img_label.shape[0], img_label.shape[1], img_label.shape[2], 9])
for i in xrange(img_label.shape[0]):
    for j in xrange(img_label.shape[1]):
        for k in xrange(img_label.shape[2]):
            label_in[i, j, k, img_label[i, j, k]] = 1

# convolutional neural network
trainbatch = 20

# conv layer 1
# parameter
w_conv1_a = weight_variable([7, 7, 1, 10])
w_conv1_b = weight_variable([7, 7, 2, 6])
b_conv1 = bias_variable([16])
# image input
x_imageA = tf.placeholder(tf.float32, [trainbatch, 240, 320, 3])
x_imageB = tf.placeholder(tf.float32, [trainbatch, 120, 160, 3])
x_imageC = tf.placeholder(tf.float32, [trainbatch, 60, 80, 3])
# layer node
# image class A
x_imageA_a = tf.slice(x_imageA, [0, 0, 0, 0], [-1, -1, -1, 1])
x_imageA_b = tf.slice(x_imageA, [0, 0, 0, 1], [-1, -1, -1, -1])
h_convA1 = tf.concat(3, [conv2d(x_imageA_a, w_conv1_a), conv2d(x_imageA_b, w_conv1_b)]) + b_conv1
h_tanhA1 = tf.tanh(h_convA1)
h_poolA1 = max_pool_2x2(h_tanhA1)
# image class B
x_imageB_a = tf.slice(x_imageB, [0, 0, 0, 0], [-1, -1, -1, 1])
x_imageB_b = tf.slice(x_imageB, [0, 0, 0, 1], [-1, -1, -1, -1])
h_convB1 = tf.concat(3, [conv2d(x_imageB_a, w_conv1_a), conv2d(x_imageB_b, w_conv1_b)]) + b_conv1
h_tanhB1 = tf.tanh(h_convB1)
h_poolB1 = max_pool_2x2(h_tanhB1)
# image class C
x_imageC_a = tf.slice(x_imageC, [0, 0, 0, 0], [-1, -1, -1, 1])
x_imageC_b = tf.slice(x_imageC, [0, 0, 0, 1], [-1, -1, -1, -1])
h_convC1 = tf.concat(3, [conv2d(x_imageC_a, w_conv1_a), conv2d(x_imageC_b, w_conv1_b)]) + b_conv1
h_tanhC1 = tf.tanh(h_convC1)
h_poolC1 = max_pool_2x2(h_tanhC1)

# conv layer 2
# parameter
w_conv2 = weight_variable([7, 7, 16, 32])
b_conv2 = bias_variable([32])
keep_prob = tf.placeholder(tf.float32)
# layer node
# image class A
h_poolA1_drop = tf.nn.dropout(h_poolA1, keep_prob)
h_convA2 = conv2d(h_poolA1_drop, w_conv2) + b_conv2
h_tanhA2 = tf.tanh(h_convA2)
h_poolA2 = max_pool_2x2(h_tanhA2)
# image class B
h_poolB1_drop = tf.nn.dropout(h_poolB1, keep_prob)
h_convB2 = conv2d(h_poolB1_drop, w_conv2) + b_conv2
h_tanhB2 = tf.tanh(h_convB2)
h_poolB2 = max_pool_2x2(h_tanhB2)
# image class C
h_poolC1_drop = tf.nn.dropout(h_poolC1, keep_prob)
h_convC2 = conv2d(h_poolC1_drop, w_conv2) + b_conv2
h_tanhC2 = tf.tanh(h_convC2)
h_poolC2 = max_pool_2x2(h_tanhC2)

# conv layer 3
# parameter
w_conv3 = weight_variable([7, 7, 32, 64])
b_conv3 = bias_variable([64])
# layer node
# image class A
h_poolA2_drop = tf.nn.dropout(h_poolA2, keep_prob)
h_convA3 = conv2d(h_poolA2_drop, w_conv3) + b_conv3
# image class B
h_poolB2_drop = tf.nn.dropout(h_poolB2, keep_prob)
h_convB3 = conv2d(h_poolB2_drop, w_conv3) + b_conv3
# image class C
h_poolC2_drop = tf.nn.dropout(h_poolC2, keep_prob)
h_convC3 = conv2d(h_poolC2_drop, w_conv3) + b_conv3

# upsampling the image
# image class A
h_upsampleA = tf.image.resize_images(h_convA3, 240, 320, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# image class B
h_upsampleB = tf.image.resize_images(h_convB3, 240, 320, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# image class C
h_upsampleC = tf.image.resize_images(h_convC3, 240, 320, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# concatenate all three class
h_upsample = tf.concat(3, [h_upsampleA, h_upsampleB, h_upsampleC])

# softmax
# parameter
w_soft = weight_variable([192, 9])
b_soft = bias_variable([9])
# layer node
h_fc = tf.reshape(h_upsample, [-1, 192])
y_out = tf.nn.softmax(tf.matmul(h_fc, w_soft) + b_soft)

# train step
# label of every pixel
y_real = tf.placeholder(tf.float32, [trainbatch, 240, 320, 9])
# cost fonction
y_in = tf.reshape(y_real, [-1, 9])
cross_entropy = -tf.reduce_sum(y_in * tf.log(y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_in,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(301):
        sel = np.random.choice(nbsel, trainbatch, replace=False)
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_imageA: inputA[sel, :, :, :], 
                                                      x_imageB: inputB[sel, :, :, :], 
                                                      x_imageC: inputC[sel, :, :, :], 
                                                      y_real: label_in[sel, :, :, :], 
                                                      keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x_imageA: inputA[sel, :, :, :], 
                                  x_imageB: inputB[sel, :, :, :], 
                                  x_imageC: inputC[sel, :, :, :], 
                                  y_real: label_in[sel, :, :, :], 
                                  keep_prob: 1.0})


