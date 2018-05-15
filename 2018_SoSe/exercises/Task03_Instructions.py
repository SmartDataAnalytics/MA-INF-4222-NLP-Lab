import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# parameters
max_fatures = 500
embed_dim = 128
lstm_out = 196
dropout = 0.1
dropout_1d = 0.4
recurrent_dropout = 0.1
random_state = 1324
validation_size = 1000
batch_size = 16
epochs=2
verbose= 2

df = pd.read_csv('dataset_sentiment.csv')
df = df[['text','sentiment']]
print(df[0:10])

df = df[df.sentiment != "Neutral"]
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: x.replace('rt',' '))
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    
tok = Tokenizer(num_words=max_fatures, split=' ')
tok.fit_on_texts(df['text'].values)
X = tok.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

nn = Sequential()
nn.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
nn.add(SpatialDropout1D(dropout_1d))
nn.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=recurrent_dropout))
nn.add(Dense(2, activation='softmax'))
nn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(nn.summary())

Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = random_state)
nn.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose=verbose)

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score, accuracy = nn.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (accuracy))

pos_cnt, neg_cnt, pos_ok, neg_ok = 0, 0, 0, 0
for x in range(len(X_validate)):
    result = nn.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0: neg_ok += 1
        else: pos_ok += 1
    if np.argmax(Y_validate[x]) == 0: neg_cnt += 1
    else: pos_cnt += 1

print("pos_acc", pos_ok/pos_cnt*100, "%")
print("neg_acc", neg_ok/neg_cnt*100, "%")

X2 = ['what are u going to say about that? the truth, wassock?!']
X2 = tok.texts_to_sequences(X2)
X2 = pad_sequences(X2, maxlen=26, dtype='int32', value=0)
print(X2)
print(nn.predict(X2, batch_size=1, verbose = 2)[0])
