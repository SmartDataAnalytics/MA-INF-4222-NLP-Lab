### Hybrid Class Semantic Classifier (HCSC) ###

_Main steps:
1- Preprocessing
- Read file and do segmentation ---> function make_doc_list()
- Stop word elimination (using nltk.corpus stop words) and stemming (using nltk.stem) ---> function stop_stem()
- Filter words with frequency less than 3 ---> function infreq_filter()
- Find 200 most informative words as feature set:
-- Step1: Make a bag of words of all the documents --->function bow()
-- Step2: Calculating the information gain of each word
--- 1, Find entropy of whole data set ---> function entropy()
--- 2, Split the data set into two set which one contains the assumed word and the other doesn't contain it. 
--- 3, Find the entropy of each subset and the results of their multiplication to probability of occurrence an non-occurrence of assumed word respectively. 
--- 4, IG(w) = whole entropy - p(w).entropy(set contains w) - (1- p(w)).entropy(set doesn't contain w) ---> function gain()
-- Step3: Select most informative words ---> function informative_words()
- After finding out the informative words, assume them as new B-O-W and represent documents' vectors based on that. ---> function doc_to_vec()
- Finally make list of documents with new representation ---> function make_vec_list().
** As this step is could be performed independently, informative words are found out once and saved for next tests.

2- Calculate meaning value of each word in each class ---> function word_meaning():
- Find the frequency of each word in each document and then the frequency of that word in each class
- Length of each class(B) is calculated by summing up the frequency of each word in that class
- Length of whole corpus(L) is calculated by summing up the lengths of all classes
- Then NFA of word w(with m frequency) in class Ci with respect to corpus S with length(L), is the expected value of co-occurrence of m-tuple(words) together with m.
- NFA(w)  = combination(k, m) * (N^(1-m)), which N = L/B and K is the frequency of w in whole corpus.
- And meaning value of w = (-1/m) * log(NFA(w)).
- The final result is the matrix M indicating the meaning value of each word in each class.

3- Labelling unlabelled documents:
- Based on the matrix M, d.M(inner product) indicates meaning value of document d in each class. The greatest score indicates the label of d. ---> function labeling()
- Add new labelled documents to training set.

4- Calculating words' weights in each class ---> function c_w_k():
- For each word w, find out #documents containing w (in nw list).
- For each word w, find out #documents containing w in each class (in dictionary cls_freq).
- Compute weight of w in each class:  cwk[w] = log(cls_freq[class_label][w] + 1)) * log(n/nw[w])
- The result is matrix W, showing weight of each word in each class.

5- Classifier ---> function classifier():
- Make train vector set train.W(inner product)
- Make test vector set test.W(inner product)
- Call SVC to figure out classifier with one-against-one method: 
- Do prediction for each document in test set and compute the precision:

Data set addresses:
- Train set: http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-all-terms.txt
- Test set: http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-all-terms.txt

Programming environment: Python 3.5
 
