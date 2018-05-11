from __future__ import division, print_function, absolute_import
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PyQt5
from PyQt5.QtWidgets import QApplication
import os

pyqt = os.path.dirname(PyQt5.__file__)
QApplication.addLibraryPath(os.path.join(pyqt, "qt\plugins"))
import tflearn
import tensorflow as tf
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector
from labprojecttf.trainPolarityLSTM import LSTM_model, GetPolarity
from labprojecttf.getsentimentscore import GetSentenceAverageSentimentScore
from labprojecttf.trainSubjectivityLSTM import GetSubjectivity
np.set_printoptions(linewidth = 150)

SENTIMENT_MAP_DIM = 10


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=None)

def ConstructSentimentMaps(text):
    sentiment_map = {'pos': np.zeros((SENTIMENT_MAP_DIM,SENTIMENT_MAP_DIM)), 
                     'neg':np.zeros((SENTIMENT_MAP_DIM,SENTIMENT_MAP_DIM)),
                     'so':np.zeros((SENTIMENT_MAP_DIM,SENTIMENT_MAP_DIM))}
    scores = []
    soft_pos = []
    soft_neg = []
    j, k = 0,0
    
    sentence_tokenized_text = sent_tokenize(text) 
    with tf.device('/cpu:0'):
        for sentence in sentence_tokenized_text:
            if GetSubjectivity(sentence):
                polarity = 'pos' if GetPolarity(sentence) == 1 else 'neg'
                score = GetSentenceAverageSentimentScore(sentence, polarity)
                if(score):
                    scores.append({'val': score, 'j': j, 'k': k, 'polarity':polarity})
                    sentiment_map[polarity][j, k] = score 
            else:
                sentiment_map['so'][j, k] = 1 #if not subjective, mark it as being objective
            k += 1
            if k == SENTIMENT_MAP_DIM:
                k = 0
                j += 1
        
        for score in scores:
            if score['polarity'] == 'pos':
                soft_pos.append(score['val'])
            else:
                soft_neg.append(score['val'])
                
        if len(soft_pos) != 0:        
            soft_pos = softmax(np.array(soft_pos))
        if len(soft_neg) != 0:    
            soft_neg = softmax(np.array(soft_neg))
        
        m, n = 0,0
        for score in scores:
            if score['polarity'] == 'pos':
                if len(soft_pos) != 0: 
                    sentiment_map['pos'][score['j'], score['k']] = soft_pos[m]
                    m = m + 1
            else:
                if len(soft_neg) != 0: 
                    sentiment_map['neg'][score['j'], score['k']] = soft_neg[n]
                    n = n + 1
        sentiment_map['pos'] = (255*sentiment_map['pos']).astype(int)
        sentiment_map['neg'] = (255*sentiment_map['neg']).astype(int)
        sentiment_map['so'] = (255*sentiment_map['so']).astype(int)
        return sentiment_map   


