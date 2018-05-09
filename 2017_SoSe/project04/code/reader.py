'''
	Author: geraltofrivia.
	This file reads the data and creates the input and output labels.
'''


import os
import sys
import xmltodict
from pprint import pprint

import numpy as np

# Some Macros
DATA_DIR = "data/"




file_names = [ x for x in os.listdir(DATA_DIR) if '.xml' in x ]

def resolve(key, edges):
	answer = None

	if key[0] == u'a':
		
		return u'e' + key[1:]

	for edge in edges:
		
		if key == edge[0]:
			#Then this triple has the desired key as a 'c' variable. Treat it's last value as the desired thing.
			answer = edge[2]

	if not answer[0] == 'e':
		answer = resolve(answer, edges)

	return answer

def get_data():
	# Empty lists
	X = []
	Y = []
	ac_words = 0
	ac_nums = 0

	# For every filename, fetch it and do parsing voodoo
	for file_name in file_names:
		file_data = open( os.path.join(DATA_DIR, file_name) ).read()
		file_parsed = xmltodict.parse(file_data)

		texts = file_parsed['arggraph']['edu']

		'''
			Input Labels.
			Also find 
			-> max words in a sentence (ac_words)
			-> max sentences in one data point (ac_nums)
		'''
		x = [ '' for temp in texts ]
		for text in texts:
			text_clean = text['#text'][:-1] + text['#text'][-1].replace(',','').replace('.','').replace('!','').replace('?','')
			x[int(text[u'@id'][1])-1] = text_clean
			if len(text_clean.split()) > ac_words:
				ac_words = len(text_clean.split())

		if len(texts) > ac_nums:
			ac_nums = len(texts)

		X.append(x)

	'''
		We got the X, and the ac_nums and ac_words. Now to move on to making y.
		But first, pad the X :]
	'''
	for i in range(len(X)):
		X[i] = X[i] + [ ''  for temp in range(ac_nums - len(X[i]))]


	'''
		Output Labels
	'''
	for file_name in file_names:
		file_data = open( os.path.join(DATA_DIR, file_name) ).read()
		file_parsed = xmltodict.parse(file_data)

		edges = file_parsed['arggraph']['edge']
		texts = file_parsed['arggraph']['edu']

		edges = [ (edge['@id'], edge['@src'], edge['@trg']) for edge in edges ]

		y = np.zeros((ac_nums, ac_nums))	#Takes care of padding the y
		for edge in edges:
			if edge[1][0] != 'e':
				src = edge[1]
				trg = edge[2]

				src = resolve(src, edges)
				trg = resolve(trg, edges)

				y[int(src[1])-1][int(trg[1])-1] = 1.0

		'''
			Now, the root AC has the same encoding as that of the padded parts. i.e. 0000000...0.
			According to the paper, we want it to point it to itself. So let's work on that now :]
		'''
		for i in range(len(texts)):
			if y[i].sum() == 0:
				y[i][i] = 1

		Y.append(y)

	return X, np.asarray(Y), ac_nums, ac_words


if __name__ == "__main__":
	x_data,y_data = get_data()
	print x_data[0]
	print y_data[0]