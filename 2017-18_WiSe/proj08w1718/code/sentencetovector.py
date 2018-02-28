import nltk
from nltk import StanfordNERTagger
import os
java_path = "C:\\Program Files (x86)\\Java\\jre1.8.0_151\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path
meta_path = 'C:\\Users\\seyit\\workspace\\nlp\\'
st = StanfordNERTagger(meta_path+'stanford-ner\\classifiers\\english.all.3class.caseless.distsim.crf.ser.gz', meta_path+'stanford-ner\\stanford-ner.jar')
from autocorrect import spell
from string import punctuation
import numpy as np


MAX_SENTENCE_LENGHT = 50 #words
NUM_DICTIONARY_WORDS = 466557

data_path = "C:\\Users\\seyit\\Desktop\\NLPLab\\data\\"
words_file = "words.txt"
with open(data_path + words_file, 'r') as f:
    words_dictionary = f.read().splitlines()
    
abbreviations = ['don''t', 'doesn''t', 'haven''t', 'hasn''t', 'hadn''t', 'wouldn''t', 'needn''t', 'shouldn''t', 'shan''t', 'won''t']
abbreviations_detached = ['do not', 'does not', 'have not', 'has not', 'had not', 'would not', 'need not', 'should not', 'shall not', 'will not']
    
def RemovePunctuationAndCorrectSpelling(text):
    removed = ''.join([w if w not in punctuation else ' ' for w in text])
    words = removed.split()
    return ' '.join([spell(word) for word in words])

def DetachAbbreviations(sentence):
    return [abbreviations_detached[abbreviations.index(word)] if word in abbreviations else word for word in sentence.lower().split()]

def WordToInt(word):
    word = word.lower()
    try:
        return words_dictionary.index(word)
    except ValueError:
        try:
            return words_dictionary.index(word.title())
        except ValueError:
            try:
                return words_dictionary.index(word.upper())
            except ValueError:
                return 0                

def StripNER(sentence):
    for word in nltk.sent_tokenize(sentence):
        tokens = nltk.tokenize.word_tokenize(word)
        tags = st.tag(tokens)
        nerless_sentence = [tag[0] for tag in tags if tag[1].upper() != 'PERSON' and tag[1].upper() != 'ORGANIZATION']
        nerless_sentence = ' '.join([word for word in nerless_sentence])
        return nerless_sentence                
        
def SentenceToVector(sentence):
    sentence = RemovePunctuationAndCorrectSpelling(sentence)
    sentence = [WordToInt(w) for w in sentence.split()]
    sentence = np.array(sentence)
    if len(sentence) == 0: #if the sentence consists of salt punctuation and was erroneously tokenized 
        return []
    vector = np.zeros(MAX_SENTENCE_LENGHT, dtype=int)
    vector[-len(sentence):] = sentence[:MAX_SENTENCE_LENGHT]
    return vector

def SentenceToVectorWithoutPadding(sentence):
    sentence = RemovePunctuationAndCorrectSpelling(sentence)
    sentence = [WordToInt(w) for w in sentence.split()]
    sentence = np.array(sentence)
    if len(sentence) == 0: #if the sentence consists of salt punctuation and was erroneously tokenized 
        return []
    else:
        return sentence

