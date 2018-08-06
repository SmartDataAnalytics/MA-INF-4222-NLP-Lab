#!/usr/bin/env python3
from gensim.models import doc2vec
from collections import namedtuple
from gensim.utils import to_unicode
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

"""
This is a litte helper module providing a doc2vec class
which can be thrown into a sklearn pipeline. A little bit modified taken from:
https://github.com/fanta-mnix/sklearn-doc2vec/blob/master/word_embeddings.py
"""

def documentize(X):
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(X):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    return docs

class Doc2VecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, size=300, window=8, min_count=5):
        self.size = size
        self.window = window
        self.min_count = min_count
        self._model = None

    def fit(self, X, y=None):
        model = doc2vec.Doc2Vec(documentize(X), size=self.size, window=self.window, min_count=self.min_count)

        self._model = model
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._model.docvecs

    def transform(self, X, copy=True):
        assert self._model is not None, 'model is not fitted'
        return np.array([self._model.infer_vector(document.words) for document in documentize(X)])

