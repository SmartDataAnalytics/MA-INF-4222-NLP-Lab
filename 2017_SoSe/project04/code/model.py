import numpy as np
from pprint import pprint
from unidecode import unidecode
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json, os
from keras.layers import Embedding

import reader
from ptrnet.seq2seq import cells
from keras import optimizers
from keras import metrics
from keras.callbacks import EarlyStopping

from ptrnet.seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq, Pointer
# from keras.utils.test_utils import keras_test
from pprint import pprint
import keras.backend as K
import numpy as np
from ptrnet import utils

from gensim import models, corpora
from sklearn.cross_validation import KFold




GLOVE_DIR = "/data/nilesh/glove" #https://nlp.stanford.edu/projects/glove/
WORD2VEC_DIR = "/data/nilesh/word2vec" #https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
EMBEDDING_DIM = 300
BS = 16

def shuffle_copy(a,b):
	assert len(a) == len(b)
	np.random.seed(0)
	p = np.random.permutation(len(a))
	return a[p],b[p]

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

model_word2vec = models.KeyedVectors.load_word2vec_format(os.path.join(WORD2VEC_DIR,'GoogleNews-vectors-negative300.bin'), binary=True)



''' 
	F1 measure 
'''
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return true_positives

def true_positives(y_true, y_pred):
	return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def predicted_positives(y_true, y_pred):
	return K.sum(K.round(K.clip(y_pred, 0, 1)))

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def sentence_embedding(sentence, embedding_type):
	'''
		sentence = a list of words ["The","fastest","person","on","the","planet"]
		embedding_type = GLOVE, Word2Vec
	'''
	if embedding_type == "GLOVE":
		sen_embedding = np.zeros(300,dtype=np.float32)
		for word in sentence:
			try:
				embedding = embeddings_index[word]
			except:
				embedding = np.zeros(300,dtype=np.float32)
			sen_embedding = sen_embedding + embedding
		return sen_embedding
	elif embedding_type == 'WORD2VEC':
		sen_embedding = np.zeros(300,dtype=np.float32)
		for word in sentence:
			try:
				embedding = model_word2vec[word]
			except:
				embedding = np.zeros(300,dtype=np.float32)
			sen_embedding = sen_embedding + embedding
		return sen_embedding		
			
	return None	

x, y , ac_num, ac_words = reader.get_data()
x_embedding = []
for node in x:
	node_embedding = []
	for sent in node:
		temp_sent = sent.split(" ")
		sent_embedding = sentence_embedding([temp_value.strip() for temp_value in temp_sent],"WORD2VEC")
		node_embedding.append(sent_embedding)
	x_embedding.append(node_embedding)
x_embedding = np.asarray(x_embedding)

#done with the data preperation step 
x_embedding,y = shuffle_copy(x_embedding,y)

evals = []
# 0.63
kf = KFold(x_embedding.shape[0], 5)
for i, (train_index, test_index) in enumerate(kf):
	if i == 0:
		continue
	# x_train,x_valid,x_test = x_embedding[train_index[:-BS]], x_embedding[train_index[-BS:]], x_embedding[test_index]
	# y_train,y_valid,y_test = y[train_index[:-BS]], y[train_index[-BS:]], y[test_index]

    x_train,x_test = x_embedding[train_index], x_embedding[test_index]
    y_train,y_test = y[train_index], y[test_index]

    x_train = np.concatenate([x_train,x_train[0:BS-(len(x_train)%BS)]])
    y_train = np.concatenate([y_train,y_train[0:BS-(len(y_train)%BS)]])

    x_test = np.concatenate([x_test,x_test[0:BS-(len(x_test)%BS)]])
    y_test = np.concatenate([y_test,y_test[0:BS-(len(y_test)%BS)]])

    models = Pointer(output_dim=10, hidden_dim=512, output_length=10, input_shape=(10, 300), batch_size=BS,bidirectional=True,dropout=0.3, depth=1)
    opt = optimizers.Adam()
    models.compile(loss='categorical_crossentropy', optimizer=opt,metrics = [fmeasure, metrics.binary_accuracy, metrics.categorical_accuracy])
    print models.summary()

    # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    models.fit(x_train, y_train, epochs=2000, batch_size=BS) # validation_data=(x_valid, y_valid), callbacks=[early_stopping])
    print "Done fitting model for fold ", i
    result = models.evaluate(x_test, y_test, batch_size=BS)
    evals += [result]
    print result


# y_pred = models.predict(x_test, batch_size=BS)
# print fmeasure(y_test, y_pred)
# x = []
# for node in x_data:
# 	temp = []
# 	for sent in node:
# 		temp.append([sent])
# 	x.append(temp)

# text = ''
# x_new = " ".join(["".join(node) for node in x])
# # text = " ".join(x_new)
# print x_new
	

# tokenizer = Tokenizer()
	
