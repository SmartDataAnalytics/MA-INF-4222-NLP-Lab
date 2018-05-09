from __future__ import division, print_function, absolute_import
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PyQt5
from PyQt5.QtWidgets import QApplication
import os
from os import listdir
from os.path import isfile, join
from labprojecttf.constructsentimentmaps import ConstructSentimentMaps
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


data_path_pos = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\pos\\"
data_path_neg = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\aclImdb_v1\\aclImdb\\train\\neg\\"
pos_files = [f for f in listdir(data_path_pos) if isfile(join(data_path_pos, f))]
neg_files = [f for f in listdir(data_path_neg) if isfile(join(data_path_neg, f))]

pmaps = []
nmaps = []
counter = 0

for pos_file, neg_file in zip(pos_files, neg_files):
    with open(data_path_pos + pos_file, 'r') as pf:
        ptext = pf.read()
    with open(data_path_neg + neg_file, 'r') as nf:
        ntext = nf.read()
    psentiment_map = ConstructSentimentMaps(ptext)
    nsentiment_map = ConstructSentimentMaps(ntext)
    print(counter)
    psmap = np.dstack([psentiment_map['pos'], psentiment_map['neg'], psentiment_map['so']]).flatten().astype(int)
    nsmap = np.dstack([nsentiment_map['pos'], nsentiment_map['neg'], nsentiment_map['so']]).flatten().astype(int)
    pmaps.append(psmap)
    nmaps.append(nsmap)
    counter = counter + 1
    if counter % 100 == 0:
        pmaps = np.array(pmaps)
        nmaps = np.array(nmaps)
        np.savetxt(data_path_pos +'pos_vec\\pos_vectors{}.txt'.format(counter), pmaps, fmt = '%d')
        np.savetxt(data_path_neg +'neg_vec\\neg_vectors{}.txt'.format(counter), nmaps, fmt = '%d')
        pmaps = []
        nmaps = []
