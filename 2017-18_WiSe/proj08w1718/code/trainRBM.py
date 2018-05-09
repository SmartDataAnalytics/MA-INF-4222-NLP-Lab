from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
import re
import tflearn
import tensorflow as tf
import numpy as np
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector, RemovePunctuationAndCorrectSpelling
from labprojecttf.getsentimentscore import GetSentenceAverageSentimentScore
import matplotlib.pyplot as plt
import PyQt5
from PyQt5.QtWidgets import QApplication
import os

pyqt = os.path.dirname(PyQt5.__file__)
QApplication.addLibraryPath(os.path.join(pyqt, "qt\plugins"))

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"
features_file = "combined_data\\RBMsentence_training.txt"
labels_file = "polarity_sentences_kaggle\\training.txt"

with open(data_path + labels_file, 'r', encoding="utf8") as f:
    sentences = [x for x in f.readlines()]
    labels = [x[0] for x in sentences]

labels = np.array(labels)
labels = labels.astype(float)
print(labels)
features = np.loadtxt(data_path + features_file, dtype = float)
rbm_data = np.c_[features, labels].astype(float)

RBM = BernoulliRBM(random_state=0, verbose=True)
RBM.n_components = 20
RBM.learning_rate = 0.05
RBM.n_iter = 20

MLP = MLPClassifier(activation='relu', alpha=1e-05, batch_size=10,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(100, 50), learning_rate='adaptive',
       learning_rate_init=0.01, max_iter=200, momentum=0.01,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
       warm_start=False)

logistic = linear_model.LogisticRegression()
logistic.C = 9000.0
classifier = Pipeline(steps=[('RBM', RBM), ('MLP', MLP)])

classifier.fit(features, labels)
test = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
label = classifier.predict(features)
print(label)


