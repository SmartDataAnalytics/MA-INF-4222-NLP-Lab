from __future__ import division, print_function, absolute_import
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
import pickle
import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT, NUM_DICTIONARY_WORDS,\
    SentenceToVector
np.set_printoptions(threshold=np.inf)

PROCESS = 4

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"
opinion_lexicon_positive = "opinion-lexicon-English\\positive-words.txt"
opinion_lexicon_negative = "opinion-lexicon-English\\negative-words.txt"


if PROCESS == 0:
    sentences_file = "rt-polaritydata\\rt-polaritydata\\rt-polarity_pos.txt"
    labels = np.ones((5331,), dtype = int)
    np.savetxt(data_path+'rt-polaritydata\\rt-polaritydata\\rt-polarity_pos_sentencelabels.txt', labels, fmt = '%d')
    labels = np.zeros((5331,), dtype = int)
    np.savetxt(data_path+'rt-polaritydata\\rt-polaritydata\\rt-polarity_neg_sentencelabels.txt', labels, fmt = '%d')
    
    labels_prob = np.loadtxt(data_path + 'stanford\\stanford_sentence_labels.txt', dtype = float)
    #labels_prob = np.array([x*10 for x in labels_prob])
    labels = np.array([1 if x > 0.5 else 0 for x in labels_prob], dtype = int)
    np.savetxt(data_path + 'stanford\\stanford_sentence_labels_2classes.txt', labels, fmt = '%d')
    
    
    with open(data_path + sentences_file, 'r') as f:
        sentences_raw = f.read().splitlines()
    sentences = [row.split('\t', 1)[0].strip() for row in sentences_raw]
    labels = np.array([row.split('\t', 1)[1] for row in sentences_raw], dtype='int')
    feature_vector= np.array([SentenceToVector(row) for row in sentences_raw])
    np.savetxt(data_path + 'rt-polaritydata\\rt-polaritydata\\rt-polarity_pos_sentencevectors.txt', feature_vector, fmt = '%d')
    np.savetxt(data_path + 'sentiment labelled sentences\\sentiment labelled sentences\\yelp_labelled_sentencelabels.txt', labels, fmt = '%d')

if PROCESS == 1:    
    sentences_file = "sentiment labelled sentences\\sentiment labelled sentences\\yelp_labelled.txt"
    with open(data_path + sentences_file, 'r') as f:
        sentences_raw = f.read().splitlines()
        sentences = [row.split('\t', 1)[0].strip() for row in sentences_raw]
    np.savetxt(data_path + 'sentiment labelled sentences\\sentiment labelled sentences\\yelp_sentences.txt', sentences, fmt='%s')
    
if PROCESS == 3:
    sentences_file = "combined_data\\sentences.txt"
    sentences_labels_file = "combined_data\\sentence_labels.txt"
    opinion_stripped_sentences_file = "combined_data\\opinion_stripped_sentences.txt"
    with open(data_path + opinion_lexicon_positive, 'r') as f:
        positive_words = f.read().splitlines()
    with open(data_path + opinion_lexicon_negative, 'r') as f:
        negative_words = f.read().splitlines()

    opinion_lexicon = positive_words + negative_words
    opinion_lexicon.append('not')
    
    with open(data_path + sentences_file, 'r') as f:
        sentences = f.read().splitlines()
    with open(data_path + sentences_labels_file, 'r') as f:
        labels = f.read().splitlines()
    stripped_sentences = []
    for org_sentence, label in zip(sentences, labels):
        stripped_sentence = [word for word in org_sentence.split() if word in opinion_lexicon]
        stripped_sentences.append(stripped_sentence)
    with open(data_path + opinion_stripped_sentences_file, 'w+') as f:
        for sentence in stripped_sentences:
            s = ' '.join(sentence)
            f.write(s+'\n')
if PROCESS == 4:
    sentences_file = "rotten_imdb\\subjective_sentences.txt"
    with open(data_path + sentences_file, 'r') as f:
        sentences = f.read().splitlines()
    feature_vector = np.array([SentenceToVector(row) for row in sentences])
    np.savetxt(data_path + 'rotten_imdb\\subjective_sentences_vectors.txt', feature_vector, fmt = '%d')
    labels = np.ones((5000,), dtype = int)
    np.savetxt(data_path+'rotten_imdb\\subjective_sentences_labels.txt', labels, fmt = '%d')
    
    