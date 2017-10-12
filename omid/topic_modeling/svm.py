
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import nltk
import numpy as np
from collections import Counter
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from numpy import savetxt, loadtxt

CONST_WIKI_ALL = './datasets/wiki_4classes.csv'
MODEL_PATH = './models/svm/matrix.gz'

print ('Reading the input..')
dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1,
                         dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})

dataset = np.array(dataset)
docs = dataset['text']
labels = dataset['klass']

# compile documents
print ('Preparing input..')
doc_complete = [nltk.re.sub(r'[^\x00-\x7F]+', ' ', line.strip()) for line in docs]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
print ('Creating the word dictionary..')
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
print ('Converting bag-of-words corpus..')
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]

print ('Creating the non-sparse matrix..')
numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary)).T # this is very expensive and time-consuming..

savetxt(MODEL_PATH, numpy_matrix)
# numpy_matrix = loadtxt(MODEL_PATH)


print ('Preparing the k-fold test and training data..')
X_train, X_test, y_train, y_test = train_test_split(numpy_matrix, labels, test_size=0.2, random_state=5)

print ('Training the classifier..')
svm = LinearSVC()
clf = CalibratedClassifierCV(svm)
clf.fit(X_train, y_train)

# predicts the probability of X_test to be in each of the four category
print (clf.predict_proba(X_test))

# svm.fit(X_train, y_train)
# print "Accuracy:", svm.score(X_test, y_test)
