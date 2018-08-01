
# coding: utf-8

# # Emoji Distance
# a notebook dealing witch emoji distance measures. Uses an external csv with labeled data to compare arbitriary emojis related to sentiment
# Autor = Carsten Draschner
# Version = 0.1
# ## Used Ressources
# https://www.clarin.si/repository/xmlui/handle/11356/1048
# https://github.com/words/emoji-emotion

# In[34]:


import pandas as pd
import math
import numpy as np


# In[35]:


N=3


# In[2]:


#read in csv as panda file
df = pd.read_csv("../Tools/Emoji_Sentiment_Data_v1.0.csv", delimiter=";")
#df.head()


# In[3]:


def dataframe_to_dictionary():
    data = {}
    data_only_emoticons = {}
    list_sentiment_vectors = []
    list_emojis = []
    list_sentiment_emoticon_vectors = []
    list_emoticon_emojis = []
    for index, row in df.iterrows():
        emo = row["Emoji"]
        occ = row["Occurrences"]
        pos = row["Positive"]
        neg = row["Negative"]
        neu = row["Neutral"]
        data.update({emo:[pos/occ,neg/occ,neu/occ]})
        
        list_sentiment_vectors.append(np.array([pos/occ,neg/occ,neu/occ]))
        list_emojis.append(emo)
        
        if(row["Unicode block"]=="Emoticons"):
            data_only_emoticons.update({emo:[pos/occ,neg/occ,neu/occ]})
            
            list_sentiment_emoticon_vectors.append(np.array([pos/occ,neg/occ,neu/occ]))
            list_emoticon_emojis.append(emo)


    return data,data_only_emoticons,np.array(list_sentiment_vectors), np.array(list_emojis), np.array(list_sentiment_emoticon_vectors),np.array(list_emoticon_emojis)
#d , doe = dataframe_to_dictionary()


# In[4]:

# create global emoji lists and datasets
data , data_only_emoticons, list_sentiment_vectors , list_emojis , list_sentiment_emoticon_vectors , list_emoticon_emojis = dataframe_to_dictionary()


# In[5]:


#calculates vector distance between 2 3-dim sentiment representations of emojis
def sentiment_vector_dist(v1,v2):
    #calculates vector distance between 2 3-dim sentiment representations of emojis consisting of positive neutral and negative probabilistic occuring
    tmp_dist = np.linalg.norm(np.array(v1)-np.array(v2))  
    return tmp_dist


# In[6]:


#calculates vector representation in a 3dim 0 to 1space of dimension: positive,negative,neutral
def emoji_to_sentiment_vector(e, only_emoticons=True):
    """tmp = df[df["Emoji"]==e]    
    #calculate by espacial labeled occurences devided by sum of overall occurences
    pos = tmp["Positive"].values[0]/tmp["Occurrences"].values[0]
    neg = tmp["Negative"].values[0]/tmp["Occurrences"].values[0]
    neu = tmp["Neutral"].values[0]/tmp["Occurrences"].values[0]
    #return as np array
    return np.array([pos,neg,neu])"""
    if e in (data_only_emoticons if only_emoticons else data):
        return np.array((data_only_emoticons if only_emoticons else data)[e])
    return np.array([float('NaN')]*N) 


# In[7]:


#function to call for evaluating two emojis in its sentimental distance
def emoji_distance(e1,e2):
    sent_v1 = emoji_to_sentiment_vector(e1)
    sent_v2 = emoji_to_sentiment_vector(e2)
    
    d = sentiment_vector_dist(sent_v1,sent_v2)
    return d


# In[27]:


def sentiment_vector_to_emoji(v1, only_emoticons=True, custom_target_emojis=None, n_results=1):

    target_sentiment_emojis = (list_sentiment_emoticon_vectors if only_emoticons else list_sentiment_vectors)
    target_emojis = (list_emoticon_emojis if only_emoticons else list_emojis)

    # filter target emojis by custom emojis, if some are given:
    if custom_target_emojis is not None:
        binary_filter_mask = np.isin(target_emojis, custom_target_emojis)
        target_sentiment_emojis = target_sentiment_emojis[binary_filter_mask]
        target_emojis = target_emojis[binary_filter_mask]

    #more efficient approach for min distance
    distances = target_sentiment_emojis - v1
    distances = np.linalg.norm(distances, axis=1)
    #find min entry
    sorted_entrys = np.argsort(distances)
    min_entry = np.argmin(distances)
    
    #print(distances[sorted_entrys[:n_results]])
    return target_emojis[min_entry] if n_results == 1 else target_emojis[sorted_entrys[:n_results]]

    #version for dics

    """#set initial values to compare with
    best_emoji = "😐"
    min_distance = 10000

    #compare only with filtred emoticons not containing other elements like cars etc.
    #compare for each existing emoticons sentment vector to find the minimal distance equivalent to the best match
    for e,v2 in doe.items():
        #v2 = emoji_to_sentiment_vector(e)
        d = sentiment_vector_dist(v1,v2)
        if(d < min_distance):
            min_distance = d
            best_emoji = e


    #print("for sentiment vector: "+str(v1)+" the emoji is : "+str(best_emoji)+" with distance of "+str(min_distance)+"!")
    return best_emoji"""

    #old version

    """#set initial values to compare with
    best_emoji = "😐"
    min_distance = 10000

    #compare only with filtred emoticons not containing other elements like cars etc.
    df_filtered = df[df["Unicode block"]=="Emoticons"]
    all_smilies = list(df_filtered["Emoji"])
    #compare for each existing emoticons sentment vector to find the minimal distance equivalent to the best match
    for e in all_smilies:
        v2 = emoji_to_sentiment_vector(e)
        d = sentiment_vector_dist(v1,v2)
        if(d < min_distance):
            min_distance = d
            best_emoji = e


    #print("for sentiment vector: "+str(v1)+" the emoji is : "+str(best_emoji)+" with distance of "+str(min_distance)+"!")
    return best_emoji"""


# In[28]:


def show_demo_min_distances(only_emoticons = True):
    #df_filtered = df[df["Unicode block"]=="Emoticons"]
    all_smilies = list_emoticon_emojis if only_emoticons else list_emojis

    d_m = np.zeros(shape=(len(all_smilies),len(all_smilies)))

    for c1 in range(len(all_smilies)):
        for c2 in range(len(all_smilies)):
            e1 = all_smilies[c1]
            e2 = all_smilies[c2]

            d = emoji_distance(e1,e2)
            d_m[c1,c2] = d
            
    for c in range(len(d_m[0])):
        emoji = all_smilies[c]
        row = d_m[c]
        row_sorted = np.argsort(row)
        #closest 5
        r = row_sorted[0:10]
        #print()
        closest = ""
        for i in r:
            closest+=all_smilies[i]
        print(emoji+": "+closest)
    
    """df_filtered = df[df["Unicode block"]=="Emoticons"]
    all_smilies = list(df_filtered["Emoji"])

    d_m = np.zeros(shape=(len(all_smilies),len(all_smilies)))

    for c1 in range(len(all_smilies)):
        for c2 in range(len(all_smilies)):
            e1 = all_smilies[c1]
            e2 = all_smilies[c2]

            d = emoji_distance(e1,e2)
            d_m[c1,c2] = d
            
    for c in range(len(d_m[0])):
        emoji = all_smilies[c]
        row = d_m[c]
        row_sorted = np.argsort(row)
        #closest 5
        r = row_sorted[0:10]
        #print()
        closest = ""
        for i in r:
            closest+=all_smilies[i]
        print(emoji+": "+closest)"""


# In[29]:


#show_demo_min_distances()


# In[30]:


#test bipolar matching entiment vector vs. emoji
#def show_demo_matching_bipolar
#    df_filtered = df[df["Unicode block"]=="Emoticons"]
#    all_smilies = list(df_filtered["Emoji"])
#    for e in all_smilies:
#        v2 = emoji_to_sentiment_vector(e)
#        sentiment_vector_to_emoji(v2)


# In[36]:


#[(e,sentiment_vector_to_emoji(emoji_to_sentiment_vector(e,only_emoticons=False))) for e in list_emojis]


# In[26]:


#sentiment_vector_to_emoji(np.array([ 0.72967448,  0.05173769,  0.21858783]))

