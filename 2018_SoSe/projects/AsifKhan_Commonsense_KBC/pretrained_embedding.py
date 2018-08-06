from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import multiprocessing
import sys
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


def word2vec_omcs(args):
        input_file = args.corpus_file
        output_file = args.output_file
        embedding = args.embedding_size
	window = args.window_size
	sg = args.sg
        iterations = args.iter

        model = Word2Vec(LineSentence(input_file), size=embedding, sg=sg, window=window, min_count=1, iter=iterations, workers=multiprocessing.cpu_count())
        model.save_word2vec_format(output_file, binary=False)


def main():
        parser = ArgumentParser('Word2vec', formatter_class = ArgumentDefaultsHelpFormatter,conflict_handler = 'resolve')
        parser.add_argument('--corpus_file', help='File with Text Corpus')
        parser.add_argument('--output_file', default='embeddings.txt', help='output file-name')
        parser.add_argument('--embedding_size', default=200, type=int, help='Embedding size')
        parser.add_argument('--iter', default=20, type=int, help='Number of iterations')
        parser.add_argument('--window_size', default=5, type=int, help='Window Size')
        parser.add_argument('--sg', type=int, default=0, help = 'Training Algorithm 0:CBOW 1:Skipgram')
		
	args = parser.parse_args()
        word2vec_omcs(args)

if __name__ == '__main__':
	sys.exit(main())
