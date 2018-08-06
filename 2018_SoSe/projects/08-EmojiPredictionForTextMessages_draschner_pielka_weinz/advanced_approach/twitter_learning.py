
# coding: utf-8

# In[1]:


import pandas as pd
from IPython.display import clear_output, Markdown, Math
import ipywidgets as widgets
import os
import glob
import json
import numpy as np
import itertools
import sklearn.utils as sku
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.cluster import KMeans
import nltk
from keras.models import load_model
from sklearn.externals import joblib
import pickle
import operator
from sklearn.pipeline import Pipeline
import json
import datetime
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from keras import losses

# check whether the display function exists:
try:
    display
except NameError:
    print("no fancy display function found... using print instead")
    display = print

# In[2]:


import sys
sys.path.append("..")

import Tools.Emoji_Distance as edist
import Tools.sklearn_doc2vec as skd2v

def emoji2sent(emoji_arr, only_emoticons=True):
    return np.array([edist.emoji_to_sentiment_vector(e, only_emoticons=only_emoticons) for e in emoji_arr])

def sent2emoji(sent_arr, custom_target_emojis=None, only_emoticons=True):
    return [edist.sentiment_vector_to_emoji(s, custom_target_emojis=custom_target_emojis, only_emoticons=only_emoticons) for s in sent_arr]

# In[3]:

SINGLE_LABEL = True

# top 20 emojis:
top_20 = list("😳😋😀😌😏😔😒😎😢😅😁😉🙌🙏😘😊😩😍😭😂")
top_20_sents = emoji2sent(top_20)

# plotting function to evaluate stuff:
def sentiment_score(s):
    #(pos, neg, neu)^T
    return s[0] - s[1]

def plot_sentiment_space(predicted_sentiment_vectors, top_sentiments, top_emojis, style='bo', additional_patches = None):
    # sentiment score axis
    top_X = np.array([sentiment_score(x) for x in top_sentiments])
    pred_X = np.array([sentiment_score(x) for x in predicted_sentiment_vectors])

    # neutral axis:
    top_Y = np.array([x[2] for x in top_sentiments])
    pred_Y = np.array([x[2] for x in predicted_sentiment_vectors])

    fig_1, ax_1 = plt.subplots()#figsize=(15,10))
    plt.title("sentiment-score-plot")
    plt.xlabel("sentiment score")
    plt.ylabel("neutrality")
    plt.xlim([-1,1])
    plt.ylim([0,1])
    for i in range(len(top_X)):
        plt.text(top_X[i], top_Y[i], top_emojis[i])
    plt.plot(pred_X, pred_Y, style)
    for p_tuple in additional_patches:
        ax_1.add_artist(p_tuple[0])
        p_tuple[0].set_alpha(0.4)
    plt.savefig("val-error_sentiment-plot" + str(datetime.datetime.now()) +  ".png", bbox_inches='tight')

    # sentiment score axis
    top_X = np.array([x[0] for x in top_sentiments])
    pred_X = np.array([x[0] for x in predicted_sentiment_vectors])

    # neutral axis:
    top_Y = np.array([x[1] for x in top_sentiments])
    pred_Y = np.array([x[1] for x in predicted_sentiment_vectors])

    fig_2, ax_2 = plt.subplots()#figsize=(15,10))
    plt.title("positive-negative-plot")
    plt.xlabel("positive")
    plt.ylabel("negative")
    plt.xlim([0,1])
    plt.ylim([0,1])
    for i in range(len(top_X)):
        plt.text(top_X[i], top_Y[i], top_emojis[i])
    plt.plot(pred_X, pred_Y, style)
    for p_tuple in additional_patches:
        ax_2.add_artist(p_tuple[1])
        p_tuple[1].set_alpha(0.4)
    plt.savefig("val-error_positive-negative-plot" + str(datetime.datetime.now()) + ".png", bbox_inches='tight')
    plt.show()

# ----
# ## classes and functions we are using later:
# ----

# * functions for selecting items from a set / list

# In[4]:


def latest(lst):
    return lst[-1] if len(lst) > 0 else 'X' 
def most_common(lst):
    # trying to find the most common used emoji in the given lst
    return max(set(lst), key=lst.count) if len(lst) > 0 else "X" # setting label to 'X' if there is an empty emoji list


# * our emoji blacklist (skin and sex modifiers)

# In[5]:


# defining blacklist for modifier emojis:
emoji_blacklist = set([
    chr(0x1F3FB),
    chr(0x1F3FC),
    chr(0x1F3FD),
    chr(0x1F3FE),
    chr(0x1F3FF),
    chr(0x2642),
    chr(0x2640)
])


# * lemmatization helper functions

# In[6]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# global stemmer and lemmatizer function
stemmer = SnowballStemmer("english")

def stem(s):
    stemmed_sent = []
    for word in s.split(" "):
        word_stemmed = stemmer.stem(word)
        stemmed_sent.append(word_stemmed)
    stemmed_sent = (" ").join(stemmed_sent)
    return stemmed_sent


lemmatizer = WordNetLemmatizer()

def lemm(s):
    lemmatized_sent = []
    sent_pos = pos_tag(word_tokenize(s))
    for word in sent_pos:
        wordnet_pos = get_wordnet_pos(word[1].lower())
        word_lemmatized = lemmatizer.lemmatize(word[0], pos=wordnet_pos)
        lemmatized_sent.append(word_lemmatized)
    lemmatized_sent = (" ").join(lemmatized_sent)
    return lemmatized_sent


def batch_stem(sentences):
    return [stem(s) for s in sentences]

def batch_lemm(sentences):
    return [lemm(s) for s in sentences]


# ### sample data manager
# the sample data manager loads and preprocesses data
# most common way to use:
# 
# 
# * `sdm = sample_data_manager.generate_and_read(path:str, only_emoticons=True, apply_stemming=True, n_top_emojis=-1, file_range=None)`
# 
#     * Generates a sample_data_manager object and preprocess data in one step
# 

# In[7]:


class sample_data_manager(object):
    @staticmethod
    def generate_and_read(path:str, only_emoticons=True, apply_stemming=True, n_top_emojis=-1, file_range=None, n_kmeans_cluster=-1, read_progress_callback=None, stem_progress_callback=None, emoji_mean=False, custom_target_emojis = None, min_words=0):
        """
        generate, read and process train data in one step.
        
        @param path: folder containing json files to process
        @param only_emoticons: if True, only messages containing emoticons (provided by Tools.Emoji_Distance) are used
        @param apply_stemming: apply stemming and lemmatization on dataset
        @param n_top_emojis: only use messages containing one of <`n_top_emojis`>-top emojis. set to `-1` to prevent top emoji filtering
        @param file_range: range of file's indices to read (eg `range(3)` to read the first three files). If `None`: all files are read
        @param n_kmeans_cluster: generating multilabeled labels with kmeans with these number of clusters. Set to -1 to use the plain sentiment space as label
        
        @return: sample_data_manager object
        """
        sdm = sample_data_manager(path)
        sdm.read_files(file_index_range=range(sdm.n_files) if file_range is None else file_range, only_emoticons=only_emoticons, progress_callback=read_progress_callback, emoji_mean=emoji_mean)
        if apply_stemming:
            sdm.apply_stemming_and_lemmatization(progress_callback=stem_progress_callback)
        
        sdm.generate_emoji_count_and_weights()
        
        if custom_target_emojis is not None:
            sdm.filter_by_emoji_list(custom_target_emojis)

        elif n_top_emojis > 0:
            sdm.filter_by_top_emojis(n_top=n_top_emojis)
        
        if n_kmeans_cluster > 0:
            sdm.generate_kmeans_binary_label(only_emoticons=only_emoticons, n_clusters=n_kmeans_cluster)
        
        if min_words > 0:
            sdm.filter_by_sentence_length(min_words=min_words)


        return sdm
        
    
    def __init__(self, data_root_folder:str):
        """
        constructor for manual initialization
        
        @param data_root_folder: folder containing json files to process
        """
        self.data_root_folder = data_root_folder
        self.json_files = sorted(glob.glob(self.data_root_folder + "/*.json"))
        self.n_files = len(self.json_files)
        self.emojis = None
        self.plain_text = None
        self.labels = None
        self.emoji_count = None
        self.emoji_weights = None
        self.X = None
        self.y = None
        self.Xt = None
        self.yt = None
        self.top_emojis = None
        self.binary_labels = None
        self.use_binary_labels = False
        self.kmeans_cluster = None
        self.label_binarizer = None
        self.use_stemming = False
        self.use_lemmatization = False
    
    def read_files(self, file_index_range:list, only_emoticons=True, emoji_mean=False ,progress_callback=None):
        """
        reading (multiple) files to one panda table.
        
        @param file_index_range: range of file's indices to read (eg `range(3)` to read the first three files)
        @param only_emoticons: if True, only messages containing emoticons (aka smileys) are used. This classification is derived from Tools.Emoji_Distance
        @param emoji_mean: if True, using mean of all emojis instead of the last one
        """
        assert np.min(file_index_range) >= 0 and np.max(file_index_range) < self.n_files
        n = len(file_index_range)

        for i in file_index_range:
            print("reading file: " + self.json_files[i] + "...")
            raw_data_i = pd.read_json(self.json_files[i], encoding="utf-8")
            emojis_i = raw_data_i['EMOJI']
            plain_text_i = raw_data_i['text']

             # replacing keywords. TODO: maybe these information can be extracted and used
            plain_text_i = plain_text_i.str.replace("(<EMOJI>|<USER>|<HASHTAG>)","").str.replace("[" + "".join(list(emoji_blacklist)) + "]","")
            
            # filter empty labels
            empty_labels = []
            
            for e in emojis_i:
                if len(e) < 1:
                    empty_labels.append(True)
                else:
                    empty_labels.append(False)
                    
            empty_labels = np.array(empty_labels, dtype=np.bool_)
            
            plain_text_i = plain_text_i[np.invert(empty_labels)]
            emojis_i = emojis_i[np.invert(empty_labels)]
            
            print("ignored " + str(np.sum(empty_labels)) + " empty labels")

            if not emoji_mean:
                # so far filtering for the latest emoji. TODO: maybe there are also better approaches
                labels_i = emoji2sent([latest(e) for e in emojis_i], only_emoticons=only_emoticons )
            else:
                tmp = [np.nanmean(emoji2sent(e, only_emoticons=only_emoticons), axis=0, dtype=float) for e in emojis_i]
                c = 0
                for t in tmp:
                    # only to find and debug wrong formatted data
                    if str(type(t)) != "<class 'numpy.ndarray'>":
                        print(t, type(t))
                        print(emojis_i[c])
                        print(emoji2sent(emojis_i[c], only_emoticons=only_emoticons))
                    c += 1

                labels_i = np.array(tmp, dtype=float)

            # and filter out all samples we have no label for:
            wrong_labels = np.isnan(np.linalg.norm(labels_i, axis=1))
            labels_i = labels_i[np.invert(wrong_labels)]
            plain_text_i = plain_text_i[np.invert(wrong_labels)]
            emojis_i = emojis_i[np.invert(wrong_labels)]
            print("imported " + str(len(labels_i)) + " samples")

            if self.labels is None:
                self.labels = labels_i
            else:
                self.labels = np.append(self.labels, labels_i, axis=0)
            
            if self.emojis is None:
                self.emojis = emojis_i
            else:
                self.emojis = pd.concat([self.emojis,emojis_i],ignore_index=True)
            
            if self.plain_text is None:
                self.plain_text = plain_text_i
            else:
                self.plain_text = pd.concat([self.plain_text,plain_text_i],ignore_index=True)

            if progress_callback is not None:
                progress_callback((i+1)/n)
        
    
    def apply_stemming_and_lemmatization(self, progress_callback = None):
        """
        apply stemming and lemmatization to plain text samples
        """
        self.use_stemming = True
        self.use_lemmatization = True
        print("apply stemming and lemmatization...")
        stemmer = SnowballStemmer("english")
        n = self.plain_text.shape[0] * 2 # 2 for loops
        i = 0
        for key in self.plain_text.keys():
            stemmed_sent = []
            for word in self.plain_text[key].split(" "):
                word_stemmed = stemmer.stem(word)
                stemmed_sent.append(word_stemmed)
            stemmed_sent = (" ").join(stemmed_sent)
            self.plain_text[key] = stemmed_sent
            i += 1
            if progress_callback is not None and i % 1024 == 0:
                progress_callback(i / n)
                

            
        lemmatizer = WordNetLemmatizer()
        for key in self.plain_text.keys():
            lemmatized_sent = []
            sent_pos = pos_tag(word_tokenize(self.plain_text[key]))
            for word in sent_pos:
                wordnet_pos = get_wordnet_pos(word[1].lower())
                word_lemmatized = lemmatizer.lemmatize(word[0], pos=wordnet_pos)
                lemmatized_sent.append(word_lemmatized)
            lemmatized_sent = (" ").join(lemmatized_sent)
            self.plain_text[key] = lemmatized_sent
            i += 1
            if progress_callback is not None and i % 1024 == 0:
                progress_callback(i / n)
        print("stemming and lemmatization done")
    
    def generate_emoji_count_and_weights(self):
        """
        counting occurences of emojis
        """
        self.emoji_count = {}
        for e_list in self.emojis:
            for e in set(e_list):
                if e not in self.emoji_count:
                    self.emoji_count[e] = 0
                self.emoji_count[e] += 1
        
        emoji_sum = sum([self.emoji_count[e] for e in self.emoji_count])

        self.emoji_weights = {}
        for e in self.emoji_count:
            # tfidf for emojis
            self.emoji_weights[e] = np.log((emoji_sum / self.emoji_count[e]))

        weights_sum= sum([self.emoji_weights[x] for x in self.emoji_weights])

        # normalize:
        for e in self.emoji_weights:
            self.emoji_weights[e] = self.emoji_weights[e] / weights_sum

        self.emoji_weights['X'] = 0  # dummy values
        self.emoji_count['X'] = 0

        # dump count data to json:
        f = open("count_from_read_progress_" + str(datetime.datetime.now()) + ".json", 'w')
        f.write(json.dumps(self.emoji_count, ensure_ascii=False))
        f.close()

    
    def get_emoji_count(self):
        """
        @return: descending list of tuples in form (<emoji as character>, <emoji count>) 
        """
        assert self.emoji_count is not None
        
        sorted_emoji_count = list(reversed(sorted(self.emoji_count.items(), key=operator.itemgetter(1))))
        #display(sorted_emoji_count)
        return sorted_emoji_count
    
    def filter_by_top_emojis(self,n_top = 20):
        """
        filter out messages not containing one of the `n_top` emojis
        
        @param n_top: number of top emojis used for filtering
        """
        assert self.labels is not None # ← messages are already read in
        
        self.top_emojis = [x[0] for x in self.get_emoji_count()[:n_top]]
        in_top = [edist.sentiment_vector_to_emoji(x) in self.top_emojis for x in self.labels]
        self.labels = self.labels[in_top]
        self.plain_text = self.plain_text[in_top]
        self.emojis = self.emojis[in_top]
        print("remaining samples after top emoji filtering: ", len(self.labels))
    
    def filter_by_emoji_list(self, custom_target_emojis):

        assert self.labels is not None

        in_list = [edist.sentiment_vector_to_emoji(x) in custom_target_emojis for x in self.labels]
        self.labels = self.labels[in_list]
        self.plain_text = self.plain_text[in_list]
        self.emojis = self.emojis[in_list]
        print("remaining samples after custom emoji filtering: ", len(self.labels))

    def filter_by_sentence_length(self, min_words):
        assert self.plain_text is not None

        is_long = [True if len(x.split()) >= min_words else False for x in self.plain_text]

        self.labels = self.labels[is_long]
        self.plain_text = self.plain_text[is_long]
        self.emojis = self.emojis[is_long]

        print("remaining samples after sentence length filtering: ", len(self.labels))

    def generate_kmeans_binary_label(self, only_emoticons=True, n_clusters=5):
        """
        generate binary labels using kmeans.
        
        @param only_emoticons: set whether we're using the full emoji set or only emoticons
        @param n_clusters: number of cluster we're generating in emoji's sentiment space
        """
        assert self.labels is not None
        array_sentiment_vectors = edist.list_sentiment_emoticon_vectors if only_emoticons else edist.list_sentiment_vectors
        array_sentiment_vectors = np.array(array_sentiment_vectors)
        
        list_emojis = edist.list_emoticon_emojis if only_emoticons else edist.list_emojis
        self.use_binary_labels = True
        print("clustering following emojis: " + "".join(list_emojis) + "...")
        self.kmeans_cluster = KMeans(n_clusters=n_clusters).fit(array_sentiment_vectors)
        print("clustering done")
        self.label_binarizer = LabelBinarizer()
        
        multiclass_labels = self.kmeans_cluster.predict(self.labels)
        
        # FIXME: we have to guarantee that in every dataset all classes occur.
        # otherwise batch fitting is not possible!
        # (or we have to precompute the mlb fitting process somewhere...)
        self.binary_labels = self.label_binarizer.fit_transform(multiclass_labels)
        
    
    def create_train_test_split(self, split = 0.1, random_state = 4222):
        assert self.plain_text is not None and self.labels is not None
        if self.X is not None:
            sys.stderr.write("WARNING: overwriting existing train/test split \n")
        
        labels = self.binary_labels if self.use_binary_labels else self.labels
        assert labels is not None
        self.X, self.Xt, self.y, self.yt = train_test_split(self.plain_text, labels, test_size=split, random_state=random_state)



# * the pipeline manager saves and stores sklearn pipelines. Keras models are handled differently, so the have to be named explicitly during save and load operations

# In[8]:


class pipeline_manager(object):
    @staticmethod
    def load_from_pipeline_file(pipeline_file:str):
        """
        loading a json configuration file and using it's paramters to call 'load_pipeline_from_files'
        """
        with open(pipeline_file, 'r') as f:
            d = json.load(f)
        
        keras_models = d['keras_models']
        all_models = d['all_models']
        
        return pipeline_manager.load_pipeline_from_files(pipeline_file.rsplit('.',1)[0], keras_models, all_models)


    @staticmethod
    def load_pipeline_from_files(file_prefix:str, keras_models = [], all_models = []):
        """
        load a pipeline from files. A pipeline should be represented by multiple model files in the form '<file_prefix>.<model_name>'
        
        @param file_prefix: basename of all files (without extension)
        @param keras_models: list of keras models (keras model files, only extension name). Leave this list empty if this is not a keras pipeline
        @param all_models: list of all models (including keras_models, only extension name).
        
        @return a pipeline manager object
        """
        
        pm = pipeline_manager(keras_models=keras_models)
        pm.load(file_prefix, all_models)
        return pm
    
    @staticmethod
    def create_keras_pipeline_with_vectorizer(vectorizer, layers, sdm:sample_data_manager, loss=None, optimizer=None, fit_vectorizer=True):
        '''
        creates pipeline with vectorizer and keras classifier
        
        @param vectorizer: Vectorizer object. will be fitted with data provided by sdm
        @param layers: list of keras layers. One keras layer is a tuple in form: (<#neurons:int>, <activation_func:str>)
        @param sdm: sample data manager to get data for the vectorizer
        @param loss: set keras loss function. Depending whether sdm use multiclass labels `categorical_crossentropy` or `mean_squared_error` is used as default
        @param optimizer: set keras optimizer. Depending whether sdm use multiclass labels `sgd` or `adam` is used as default

        @return: a pipeline manager object
        
        '''
        from keras.models import Sequential
        from keras.layers import Dense
        
        if fit_vectorizer:
            if sdm.X is None:
                sdm.create_train_test_split()
            
            print("fit vectorizer...")
            vec_train = vectorizer.fit_transform(sdm.X)
            vec_test = vectorizer.transform(sdm.Xt)
            print("fitting done")
        # creating keras model:
        model=Sequential()
        
        keras_layers = []
        first_layer = True
        for layer in layers:
            if first_layer:
                size = None
                if "size" in dir(vectorizer):
                    size = vectorizer.size
                else:
                    size = vectorizer.transform([" "])[0]._shape[1]
                model.add(Dense(units=layer[0], activation=layer[1], input_dim=size))
                first_layer = False
            else:
                model.add(Dense(units=layer[0], activation=layer[1]))
        
        if sdm.use_binary_labels: 
            loss_function = loss if loss is not None else 'categorical_crossentropy'
            optimizer_function = optimizer if optimizer is not None else 'sgd'
            model.compile(loss=loss_function,
                          optimizer=optimizer_function,
                          metrics=['accuracy'])
        else:
            loss_function = loss if loss is not None else 'mean_squared_error'
            optimizer_function = optimizer if optimizer is not None else 'adam'
            model.compile(loss=loss_function,
                          optimizer=optimizer_function)
        
        pipeline = Pipeline([
            ('vectorizer',vectorizer),
            ('keras_model', model)
        ])
        
        return pipeline_manager(pipeline=pipeline, keras_models=['keras_model'])
    
    @staticmethod
    def create_pipeline_with_classifier_and_vectorizer(vectorizer, classifier, sdm:sample_data_manager = None):
        '''
        creates pipeline with vectorizer and non-keras classifier
        
        @param vectorizer: Vectorizer object. will be fitted with data provided by sdm
        @param classifier: unfitted classifier object (should be compatible with all sklearn classifiers)
        @param sdm: sample data manager to get data for the vectorizer
        
        @return: a pipeline manager object
        '''
        if sdm is not None:
            if sdm.X is None:
                sdm.create_train_test_split()

            vec_train = vectorizer.fit_transform(sdm.X)
            vec_test = vectorizer.transform(sdm.Xt)
        
        pipeline = Pipeline([
            ('vectorizer',vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline_manager(pipeline=pipeline, keras_models=[])
    
    def __init__(self, pipeline = None, keras_models = []):
        """
        constructor
        
        @param pipeline: a sklearn pipeline
        @param keras_models: list of keras steps in pipeline. Neccessary because saving and loading from keras models differs from the scikit ones
        """
        
        self.pipeline = pipeline
        self.additional_objects = {}
        self.keras_models = keras_models
    
    def save(self, prefix:str):
        """
        saving the pipeline. It generates one file per model in the form: '<prefix>.<model_name>'
        
        @param prefix: file prefix for all models
        """
        

        print(self.keras_models)
        # doing this like explained here: https://stackoverflow.com/a/43415459
        for step in self.pipeline.named_steps:
            if step in self.keras_models:
                self.pipeline.named_steps[step].model.save(prefix + "." + step)
            else:
                joblib.dump(self.pipeline.named_steps[step], prefix + "." + str(step))
        
        load_command = "pipeline_manager.load_pipeline_from_files( '"
        load_command += prefix + "', " + str(self.keras_models) + ", "
        load_command += str(list(self.pipeline.named_steps.keys())) + ")"

        with open(prefix + '.pipeline', 'w') as outfile:
            json.dump({'keras_models': self.keras_models, 'all_models': [step for step in self.pipeline.named_steps]}, outfile)
        
        import __main__ as main
        if not hasattr(main, '__file__'):
            display("saved pipeline. It can be loaded the following way:")
            display(Markdown("> ```\n"+load_command+"\n```"))              # ← if we're in jupyter, print the fancy way :)
        else:
            print("saved pipeline. It can be loaded the following way:")
            print(load_command)
        
    
    def load(self, prefix:str, models = []):
        """
        load a pipeline. A pipeline should be represented by multiple model files in the form '<prefix>.<model_name>'
        NOTE: keras model names (if there are some) have to be defined in self.keras_models first!
        
        @param prefix: the prefix for all model files
        @param models: model_names to load
        """
        self.pipeline = None
        model_list = []
        for model in models:
            if model in self.keras_models:
                model_list.append((model, load_model(prefix + "." + model)))
            else:
                model_list.append((model, joblib.load(prefix+"." + model)))
        self.pipeline = Pipeline(model_list)
    
    def fit(self,X,y):
        """fitting the pipeline"""
        self.pipeline.fit(X,y)
    
    def predict(self,X, use_stemming=False, use_lemmatization=False):
        """predict"""
        if use_stemming:
            X = np.array(batch_stem(X))
        if use_lemmatization:
            X = np.array(batch_lemm(X))
        return self.pipeline.predict(X)
    


# * the trainer class passes Data from the sample manager to the pipeline manager

# In[9]:

def to_dense_if_sparse(X):
    """
    little hepler function to make data dense (if it is sparse).
    is used in trainer.fit function
    """
    if "todense" in dir(X):
        return X.todense()
    return X


class trainer(object):
    def __init__(self, sdm:sample_data_manager, pm:pipeline_manager):
        """constructor"""
        self.sdm = sdm
        self.pm = pm
        self.acc = []
        self.val = []
    
    def fit(self, max_size=1000000, disabled_fit_steps=['vectorizer'], keras_batch_fitting_layer=['keras_model'], batch_size=None, n_epochs=1, progress_callback=None):
        """
        fitting data in the pipeline. Because we don't want to refit the vectorizer, the pipeline models containing the vectorizer have to be named explicitly
        
        @param max_size: don't train more examples than that number
        @param disabled_fit_steps: list of pipeline steps that we want to prevent to refit. Normally all vectorizer steps
        """
        # TODO: make batch fitting available here (eg: continous waiting for data and fitting them)
        if self.sdm.X is None:
            self.sdm.create_train_test_split()
        disabled_fits = {}
        disabled_fit_transforms = {}
        
        disabled_keras_fits = {}
        
        named_steps = self.pm.pipeline.named_steps
        
        for s in disabled_fit_steps:
            # now it gets really dirty:
            # replace fit functions we don't want to call again (e.g. for vectorizers)
            disabled_fits[s] = named_steps[s].fit
            disabled_fit_transforms[s] = named_steps[s].fit_transform
            named_steps[s].fit = lambda self, X, y=None: self
            named_steps[s].fit_transform = named_steps[s].transform
        
        if batch_size is not None:
            for k in keras_batch_fitting_layer:
                # forcing batch fitting on keras
                disabled_keras_fits[k]=named_steps[k].fit

                named_steps[k].fit = lambda X, y: named_steps[k].train_on_batch(to_dense_if_sparse(X), y) # ← why has keras no sparse support on batch progressing!?!?!
            
        if batch_size is None:
            self.acc = []
            self.val = []
            for e in range(n_epochs):
                print("epoch", e)
                self.pm.fit(X = self.sdm.X[:max_size], y = self.sdm.y[:max_size])
                pred, yt = self.test()
                mean_squared_error = ((pred - yt)**2).mean(axis=0)
                print("#" + str(e) + ": validation loss: ", mean_squared_error, "scalar: ", np.mean(mean_squared_error))
                self.val.append(np.mean(mean_squared_error))
                plot_sentiment_space(pred, top_20_sents, top_20)
            plt.figure(figsize=(10,5))
            plt.plot(self.val)
            plt.savefig("val_error" + str(datetime.datetime.now()) + ".png", bbox_inches='tight')
            plt.show()

        else:
            n = len(self.sdm.X) // batch_size
            for i in range(n_epochs):
                for j in range(n):
                    self.pm.fit(X = np.array(self.sdm.X[j*batch_size:(j+1)*batch_size]), y = np.array(self.sdm.y[j*batch_size:(j+1)*batch_size]))
                    if progress_callback is not None:
                        progress_callback(j / n)
                    pred, yt = self.test()
                    mean_squared_error = ((pred - yt)**2).mean(axis=0)
                    print("#" + str(j) + ": loss: ", mean_squared_error)

        
        # restore replaced fit functions:
        for s in disabled_fit_steps:
            named_steps[s].fit = disabled_fits[s]
            named_steps[s].fit_transform = disabled_fit_transforms[s]
        
        if batch_size is not None:
            for k in keras_batch_fitting_layer:
                named_steps[k].fit = disabled_keras_fits[k]
    
    def test(self, use_lemmatization=False, use_stemming=False, emoji_subset=None, only_test_on_valid_set = True):
        '''
        @param use_lemmatization:boolean
        @param use_stemming:boolean
        @param emoji_subset:list if given, only make predictions on samples containing one of these emojis as teacher value
        @return: prediction:list, teacher:list
        '''



        if self.sdm.X is None:
            self.sdm.create_train_test_split()

        Xt = self.sdm.Xt
        yt = self.sdm.yt

        print("original validation size: " + str(len(yt)))

        if emoji_subset is not None:

            has_emoji = np.array([True if edist.sentiment_vector_to_emoji(y) in emoji_subset else False for y in yt])
            Xt = Xt[has_emoji]
            yt = yt[has_emoji]

            print("filtered validation size: " + str(len(yt)))


        return self.pm.predict(Xt, use_lemmatization=use_lemmatization, use_stemming=use_stemming), yt

