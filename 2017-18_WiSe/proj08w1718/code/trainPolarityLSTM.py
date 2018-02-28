
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
from tflearn.datasets import imdb
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector, StripNER
from nltk.corpus import sentence_polarity
np.set_printoptions(threshold=np.inf)

TRAIN = 0

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"

if TRAIN:
    features = np.loadtxt(data_path + 'combined_data\\sentence_vectors.txt', dtype = int)
    labels = np.loadtxt(data_path + 'combined_data\\sentence_labels.txt', dtype = int)
      
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



LSTM_model = tflearn.DNN(net, tensorboard_verbose=0)

if TRAIN:
    LSTM_model.fit(trainX, trainY, validation_set=(testX, testY), n_epoch=10, shuffle=True, show_metric=True, batch_size=30)
    LSTM_model.save("model_stanford_lstm_2classes.tfl")



LSTM_model.load('model_stanford_lstm_2classes.tfl')


'''

sentence = 'djimon hounsou and turn in excellent performances'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'it was horrendous'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'I would rather sleep on my couch than go see this movie'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'There is so much wrong with it that you really just have \nto see it for yourself, although I would recommend not paying money for it. '
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'By the end of the night, hours after we left the theater, I felt totally gutted and crestfallen\n to realize what the Star Wars sequels had become, because Rian Johnson had to "Age of Ultron" the series'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'I loved it'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
sentence = 'Not only was his behavior completely uncharacteristic, \nbut his fight scene against Emo Ren was a sham and his death was utterly meaningless'
sentencev = SentenceToVector(sentence)
sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
prediction = LSTM_model.predict_label(sentencev)
print('- ', sentence, prediction[0,0])
'''
def GetPolarity(sentence):
    with tf.device('/cpu:0'):
        sentence = StripNER(sentence)
        sentencev = SentenceToVector(sentence)
        if sentencev == []:
            return False
        sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
        prediction = LSTM_model.predict_label(sentencev)
        return prediction[0,0]
