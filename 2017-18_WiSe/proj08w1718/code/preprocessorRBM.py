from nltk.tokenize import word_tokenize
import re
import numpy as np
from labprojecttf.sentencetovector import MAX_SENTENCE_LENGHT,\
    SentenceToVector, RemovePunctuationAndCorrectSpelling
from labprojecttf.trainLSTM import LSTM_model

PROCESS = 0

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"
sentences_file = "polarity_sentences_kaggle\\training.txt"
opinion_lexicon_positive = "opinion-lexicon-English\\positive-words.txt"
opinion_lexicon_negative = "opinion-lexicon-English\\negative-words.txt"

if PROCESS:
    with open(data_path + opinion_lexicon_positive, 'r') as f:
        positive_words = f.read().splitlines()
    with open(data_path + opinion_lexicon_negative, 'r') as f:
        negative_words = f.read().splitlines()
        
    with open(data_path + sentences_file, 'r', encoding="utf8") as f:
        sentences = [x for x in f.readlines()]
        labels = [x[0] for x in sentences]
        sentences = [re.sub('[0\n\t1]', '', sentence) for sentence in sentences]
    
    with open(data_path + 'combined_data\\RBMsentence_training.txt', 'w+') as f:    
        for sentence in sentences:
            sentence = RemovePunctuationAndCorrectSpelling(sentence)
            words = word_tokenize(sentence)
            sentence_polar_word_decomp = {'lstm_prediction': 0.0, 'pos':0.0, 'neg':0.0, 'total':0.0}
            sentencev = SentenceToVector(sentence)
            sentencev = sentencev.reshape(1, MAX_SENTENCE_LENGHT)
            prediction = LSTM_model.predict_label(sentencev)
            sentence_polar_word_decomp['lstm_prediction'] += prediction[0,0]
            for word in words:
                if word in opinion_lexicon_positive:
                    sentence_polar_word_decomp['pos'] += 1.0
                    sentence_polar_word_decomp['total'] += 1.0
                elif word in opinion_lexicon_negative:
                    sentence_polar_word_decomp['neg'] += 1.0
                    sentence_polar_word_decomp['total'] += 1.0
            if sentence_polar_word_decomp['total'] != 0.0:
                f.write(str(sentence_polar_word_decomp['lstm_prediction'])+'\t'+str(sentence_polar_word_decomp['pos'] / sentence_polar_word_decomp['total'])
                        +'\t'+str(sentence_polar_word_decomp['neg'] / sentence_polar_word_decomp['total'])+'\n')
            else:
                f.write(str(sentence_polar_word_decomp['lstm_prediction'])+'\t'+str(sentence_polar_word_decomp['pos'])
                        +'\t'+str(sentence_polar_word_decomp['neg'])+'\n')
        


def scale(X, eps = 0.001):
    return (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + eps)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=None)
