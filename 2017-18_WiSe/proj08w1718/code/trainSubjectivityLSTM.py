
from __future__ import division, print_function, absolute_import
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector
np.set_printoptions(threshold=np.inf)

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"

TRAIN = 0

if TRAIN:
    features = np.loadtxt(data_path + 'combined_data\\subjectivity_sentence_vectors.txt', dtype = int)
    labels = np.loadtxt(data_path + 'combined_data\\subjectivity_sentence_labels.txt', dtype = int)
      
    train_split_proportion = 0.8
    validation_split_proporiton = 0.5
    split = int(len(features) * train_split_proportion)
    trainX, testX = features[:split], features[split:]
    trainY, testY = labels[:split], labels[split:]
    #print(trainY)
    trainY = to_categorical(trainY, 2)
    testY = to_categorical(testY, 2)
    
tf.reset_default_graph()
tflearn.config.init_training_mode()

net = tflearn.input_data([None, MAX_SENTENCE_LENGHT])
net = tflearn.embedding(net, input_dim=NUM_DICTIONARY_WORDS+1, output_dim=100)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

SubjectivityLSTM_model = tflearn.DNN(net, tensorboard_verbose=0)

if TRAIN:
    SubjectivityLSTM_model.fit(trainX, trainY, validation_set=(testX, testY), n_epoch=8, shuffle=True, show_metric=True, batch_size=30)
    SubjectivityLSTM_model.save("model_subjectivity_lstm_2classes.tfl")

SubjectivityLSTM_model.load('model_subjectivity_lstm_2classes.tfl')



def GetSubjectivity(sentence):
    with tf.device('/cpu:0'):
        sentencev = SentenceToVector(sentence)
        if sentencev == []:
            return False
        sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
        prediction = SubjectivityLSTM_model.predict_label(sentencev)
        return prediction[0,0]
