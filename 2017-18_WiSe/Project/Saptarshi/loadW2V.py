# from gensim.models.keyedvectors import KeyedVectors

# b_txt = KeyedVectors.load('tmp/brown.txt', binary=False)
# mr_txt = KeyedVectors.load_word2vec_format('tmp/movie_reviews.txt', binary=False)
# t_txt = KeyedVectors.load_word2vec_format('tmp/treebank.txt', binary=False)
#
# b_bin = KeyedVectors.load_word2vec_format('tmp/brown.bin', binary=True)
# mr_bin = KeyedVectors.load_word2vec_format('tmp/movie_reviews.bin', binary=True)
# t_bin = KeyedVectors.load_word2vec_format('tmp/treebank.bin', binary=True)

from gensim.models import Word2Vec

b = Word2Vec.load('tmp/brown.bin')
mr = Word2Vec.load('tmp/movie_reviews.bin')
t = Word2Vec.load('tmp/treebank.bin')