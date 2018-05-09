from __future__ import division, print_function, absolute_import
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import tflearn
import numpy as np
import tensorflow as tf
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector
np.set_printoptions(threshold=np.inf)

TRAIN = 0

if TRAIN:
    data_path_pos = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\pos\\pos_vec\\"
    data_path_neg = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\neg\\neg_vec\\"
    
    features_pos = np.loadtxt(data_path_pos + 'pos_vectors100.txt', dtype = int)
    features_neg = np.loadtxt(data_path_neg + 'neg_vectors100.txt', dtype = int)
    
    featuresx = np.append(features_pos, features_neg)
    features = featuresx.reshape((200, 10, 10, 3))
    labels = np.append(np.ones((100,)), np.zeros((100,)))
    
    train_split_proportion = 0.9
    split = int(len(features) * train_split_proportion)
    trainX, testX = features[:split], features[split:]
    trainY, testY = labels[:split], labels[split:]
    trainY = to_categorical(trainY, 2)
    testY = to_categorical(testY, 2)

tf.reset_default_graph()
tflearn.config.init_training_mode()

net = input_data(shape=[None, 10, 10, 3])
net = conv_2d(net, 10, 3, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 20, 3, activation='relu')
net = conv_2d(net, 20, 3, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 128, activation='relu')
net = dropout(net, 0.3)
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

CNN_model = tflearn.DNN(net, tensorboard_verbose=0)
CNN_model.load("model_CNN_2classes.tfl")
if TRAIN:
    CNN_model.fit(trainX, trainY, n_epoch=50, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=5)
    CNN_model.save("model_CNN_2classes.tfl")

