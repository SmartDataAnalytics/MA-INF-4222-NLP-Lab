# naive_approach

This directory contains the functions necessary to run the Naive Approach.

Prerequisites:
 * the file [emoji_descriptions_preprocessed.csv](../Tools/emoji_descriptions_preprocessed.csv) has to be located in the specified folder [../Tools](../Tools)
 * pandas has to be installed

For testing, import [naive_approach.py](naive_approach.py) and execute the following commands:

1. `prepareData(stem, lower)`
	* preprocesses the emoji descriptions and returns a dictionary with the indexed emojis
	* parameters:
		* `stem`: Apply stemming (default=`True`)
		* `lower`: Apply lowercasing (default=`True`)
		
2. `predict(sentence, lookup, emojis_to_consider, criteria, lang, embeddings, n=10, t=0.9)`
	* evaluates an input sentence and returns a list of predicted emojis
	* parameters:
		* `sentence`: Input sentence (required parameter)
		* `lookup`: dictionary with emoji data (return value of prepareData, required parameter)
		* `emojis_to_consider`: set of emojis to include in prediction, or `"all"` (default=`"all"`)
		* `criteria`: criteria to evaluate the values of the description - message matching.
			* options: `"sum"`, `"mean"`, `"max_val"`, `"threshold"` (default: `"threshold"`)
		* `lang`: language to use (default: "eng")
		* `embeddings`: word embeddings
			* options: `"wordnet"`, `"word2Vec"`, `"fastText"`, default: `"wordnet"`
		* `n`: number of top ranked emojis to return (default=`10`)
		* `t`: threshold for the `"threshold"` criteria (default=`0.9`)