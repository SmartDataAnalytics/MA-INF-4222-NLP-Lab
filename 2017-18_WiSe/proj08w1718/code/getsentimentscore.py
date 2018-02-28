'''
Created on 17 Feb 2018

@author: seyit
'''
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from autocorrect import spell
from string import punctuation
import csv
import tensorflow as tf
from nltk.metrics.distance import edit_distance
from numba.tests.npyufunc.test_ufunc import dtype



def GetSentenceAverageSentimentScore(sentence, polarity): #polarity: either 'pos' or 'neg'
    tokenized_text = word_tokenize(sentence)
    tagged_text = nltk.pos_tag(tokenized_text)
    total_score, count = 0.0,0.0
    for taggedword in tagged_text:
        score = GetWordSentimentScore(taggedword)
        if(score):
            total_score += score[polarity]
            count += 1.0
    if total_score == 0.0:
        return False
    else:
        return total_score/count


def GetWordType(taggedword):
    a = ['JJ', 'JJR', 'JJS']
    n = ['NN', 'NNS', 'NNP']
    v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    r = ['RB', 'RBR', 'RBS']
    if taggedword[1] in a:
        return 'a'
    elif taggedword[1] in n:
        return 'n'
    if taggedword[1] in v:
        return 'v'
    elif taggedword[1] in r:
        return 'r'
    else:
        return False

def GetWordSentimentScore(taggedword):
    wordtype = GetWordType(taggedword)
    if not wordtype: #if the word type is not included in SentiSynSet
        return False #there exists no sentiment score
    wordscore = swn.senti_synsets(taggedword[0], wordtype)
    pos, neg = 0.0, 0.0
    count = 0
    for val in wordscore:
        pos = pos + val.pos_score()
        neg = neg + val.neg_score() 
        count += 1
    if(pos == 0.0 and neg == 0.0):
        return False #there exists no sentiment score
    else:
        #print('%10s' % taggedword[0],'\t', wordtype, '\t\t', (pos, neg))
        return {'pos':pos, 'neg':neg}   
'''    
sentence = "The Matrix was amazing"
print('overall score = ', GetSentenceAverageSentimentScore(sentence, 'pos'))
sentence = "I don't think I have ever seen more amazing movie than this"
print('overall score = ', GetSentenceAverageSentimentScore(sentence, 'pos'))
sentence = "This the most exciting movie I have ever seen"
print('overall score = ', GetSentenceAverageSentimentScore(sentence, 'pos'))
'''


