from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
from pprint import pprint
import string
import nltk
import numpy as np

CONST_WIKI_ALL = './datasets/wiki_4classes.csv'
MODEL_PATH = './models/lda/lda-model'
NUMBER_OF_TOPICS = 4

print ('Reading the input..')
dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1,
                         dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})

dataset = np.array(dataset)
docs = dataset['text']
labels = dataset['klass']

doc_complete = [nltk.re.sub(r'[^\x00-\x7F]+', ' ', line.strip()) for line in docs]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(unicode(lemma.lemmatize(word)) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
print ('Creating term dictionary..')
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

print ('Training the model..')
# Train the LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=NUMBER_OF_TOPICS, id2word = dictionary, passes=20)
# ldamodel = Lda.load(MODEL_PATH)
print ('Training completed!')

def show(index):
    arr = ldamodel.get_document_topics(doc_term_matrix[i])
    arr.sort(key=lambda x: x[1] * -1)
    print arr
    
pprint(ldamodel.print_topics(num_topics=NUMBER_OF_TOPICS, num_words=10))

show(1000)
show(10000)
show(15000)
show(17000)

