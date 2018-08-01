# coding: utf-8

# In[1]:


import pandas as pd
from IPython.display import clear_output, Markdown, Math
import ipywidgets as widgets
import os
import unicodedata as uni
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
import math
import pprint

from gensim.models import Word2Vec, KeyedVectors

# # Naive Approach
table = pd.read_csv('../Tools/emoji_descriptions_preprocessed.csv', delimiter = ";")

##Store table in the format:
## { index: [emoji, description]}
tableDict = {}
for index, row in table.iterrows():
    tableDict.update({index: [row['character'], row['description']]})

#######################
# Helper functions
#######################

def stemming(message):
    ps = PorterStemmer()
    words = word_tokenize(message)
    sm = []
    for w in words:
        sm.append(ps.stem(w))
    stemmed_message = (" ").join(sm)
    return stemmed_message


# * compare words to emoji descriptions
def evaluate_sentence(sentence, description_key = 'description', lang = 'eng', emojis_to_consider="all",\
                      stem=True, embeddings="wordnet"):
    # assumes there is a trained w2v model stored in the same directory!
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    if embeddings=="word2Vec":
        wv = KeyedVectors.load(str(__location__)+"/word2vec.model", mmap='r')
    elif embeddings=="fastText":
        wv = KeyedVectors.load(str(__location__)+"/fastTextVectors.kv", mmap='r')
        
    if (stem):
        sentence = stemming(sentence)
        
    tokenized_sentence = word_tokenize(sentence)
    n = len(tokenized_sentence)
    matrix_list = []
    
    for index in tableDict.keys():
        emoji_tokens = word_tokenize(tableDict[index][1])
        m = len(emoji_tokens)

        mat = np.zeros(shape=(m,n))
        for i in range(len(emoji_tokens)):
            for j in range(len(tokenized_sentence)):
                if embeddings=="wordnet":
                    syn1 = wordnet.synsets(emoji_tokens[i],lang=lang)
                    if len(syn1) == 0:
                        continue
                    w1 = syn1[0]
                    #print(j, tokenized_sentence)
                    syn2 = wordnet.synsets(tokenized_sentence[j], lang=lang)
                    if len(syn2) == 0:
                        continue
                    w2 = syn2[0]
                    val = w1.wup_similarity(w2)
                    if val is None:
                        continue
                elif (embeddings == "word2Vec" or embeddings == "fastText"):
                    try:
                        val = wv.similarity(emoji_tokens[i], tokenized_sentence[j])
                    except KeyError:
                        continue
                mat[i,j] = val
        matrix_list.append(mat)
            
    return matrix_list
    
    
###########################
#Functions to be called from main script
###########################
    

# load and preprocess data
# emojis_to_consider can be either a list or "all"
def prepareData(stem=True, lower=True):
    if(stem):
        for index in tableDict.keys():
            tableDict[index][1] = stemming(tableDict[index][1])
    if(lower):
        for index in tableDict.keys():
            tableDict[index][1] = tableDict[index][1].lower()
    
    #collect the emojis
    lookup = {}
    emoji_set = []
    for index in tableDict.keys():
        lookup[index] = tableDict[index][0]
        emoji_set.append(tableDict[index][0])

    emoji_set = set(emoji_set)
    
    return lookup

# make a prediction for an input sentence
# embeddings = ["wordnet", "word2Vec", "fastText"]
def predict(sentence, lookup, emojis_to_consider="all", criteria="threshold", lang = 'eng',\
            embeddings="wordnet", n=10, t=0.9, stem = True):

    result = evaluate_sentence(sentence, lang, emojis_to_consider=emojis_to_consider, embeddings=embeddings, stem = stem)
    
    try:
        if(criteria=="summed"):
            resultValues = [-np.sum(x) for x in result]
        elif (criteria=="max_val"):
            resultValues = [-np.max(x) for x in result]
        elif(criteria=="avg"):
            resultValues = [-np.mean(x) for x in result]
        else:
            resultValues = [-len(np.where(x>t)[0]) / (x.shape[0] * x.shape[1]) for x in result]
        indexes = np.argsort(resultValues)
        results = np.sort(resultValues)
        
        if (emojis_to_consider != "all" and type(emojis_to_consider) == list):
            indexes2 = []
            results2 = []
            for i in range(len(indexes)):
                if lookup[indexes[i]] in emojis_to_consider:
                    indexes2.append(indexes[i])
                    results2.append(results[i])
            indexes = indexes2
            results = results2
            
        indexes = indexes[0:n]
        results = results[0:n]
        
        # build a result table
        table_array = [lookup[indexes[i]] for i in range(n) ]
          
        #table_frame = pd.DataFrame(table_array, columns=[criteria, 'description'])
        
        #display(table_frame)
        
        return table_array, results
    
    except ZeroDivisionError as err:
        print("There seems to be a problem with the input format. Please enter a nonempty string")
        return [], []


#predict("I like to travel by train", description_key='description' , lang='eng')

