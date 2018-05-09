from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from flask import Flask
import urllib
import numpy as np
import json
from elasticsearch import Elasticsearch

app = Flask(__name__)
#b = Word2Vec.load('tmp/brown.bin')
#g = Word2Vec.load('tmp/GoogleNews-vectors-negative300.bin')
g = KeyedVectors.load_word2vec_format('tmp/GoogleNews-vectors-negative300.bin', binary=True) 
log = []

@app.route("/")
def hello():
    return "OK"


def ConvertVectorSetToVecAverageBased(vectorSet, ignore = []):
	if len(ignore) == 0:
		return np.mean(vectorSet, axis = 0)
	else:
		return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)


def phrase_similarity(_phrase_1, _phrase_2):
    phrase_1 = _phrase_1.split(" ")
    phrase_2 = _phrase_2.split(" ")
    vw_phrase_1 = []
    vw_phrase_2 = []
    for phrase in phrase_1:
        try:
            # print phrase
            vw_phrase_1.append(g.word_vec(phrase.lower()))
        except:
            # print traceback.print_exc()
            continue
    for phrase in phrase_2:
        try:
            vw_phrase_2.append(g.word_vec(phrase.lower()))
        except:
            continue
    if len(vw_phrase_1) == 0 or len(vw_phrase_2) == 0:
        return 0
    v_phrase_1 = ConvertVectorSetToVecAverageBased(vw_phrase_1)
    v_phrase_2 = ConvertVectorSetToVecAverageBased(vw_phrase_2)
    cosine_similarity = np.dot(v_phrase_1, v_phrase_2) / (np.linalg.norm(v_phrase_1) * np.linalg.norm(v_phrase_2))
    return cosine_similarity

@app.route('/word2vecsimilarity/<path:phrases>/<path:labels>', methods=['GET'])
def phrase_label_similarity_cd(phrases, labels):
    phrases = urllib.unquote(phrases)
    phrases = phrases.split('+')
    labels = urllib.unquote(labels)
    labels = labels.split('+')
    
    phr_lab_avg = []
        
    for phrase in phrases:
        for label in labels:
            score = phrase_similarity(phrase, label)
            phr_lab_avg.append([phrase, label, score])
    
    return str(phr_lab_avg)

  
@app.route("/<path:synonyms>/<path:properties>", methods=["GET"])
def simple(synonyms, properties):
    synonyms = urllib.unquote(synonyms)
    synonyms = synonyms.split('+')
    properties = urllib.unquote(properties)
    properties = properties.split('+')
    
    prop_syn_avg = []
    for p in properties:
        avg = 0
        for synonym in synonyms:
            try:
                s = g.similarity(p, synonym)
                # print '(', p[0], ',' , synonym[0], ') ', s
                avg += s
            except KeyError:
                log.append(KeyError)

        prop_syn_avg.append([p, avg/len(synonyms)])

    sortd = np.argsort(np.array(prop_syn_avg), 0)[:,1]

    l = []
    for s in sortd:
        l.append(prop_syn_avg[s])

    keyword = prop_syn_avg[np.argmax(np.array(prop_syn_avg), 0)[1]][0]
    
    return str(l)
    

@app.route("/phraselabel/<path:phrases>/<path:labels>", methods=["GET"])
def phrase_label_similarity_avg(phrases, labels):
    phrases = urllib.unquote(phrases)
    phrases = phrases.split('+')
    labels = urllib.unquote(labels)
    labels = labels.split('+')
    
    phr_lab_avg = []
    
    phrases_arr = []
    for phrase in phrases:
        phrases_arr.append(phrase.split(' '))
    
    labels_arr = []
    for label in labels:
        labels_arr.append(label.split(' '))
        
    for words_pa in phrases_arr:
        for words_la in labels_arr:
            avg = 0.0
            count = 0.0
            for word_pa in words_pa:
                for word_la in words_la:
                    try:
                        s = g.similarity(word_pa, word_la)
                        avg += s
                    except KeyError:
                        log.append(KeyError) 
            
                    count += 1
                    
            if count > 0:
                avg = avg/count
            
            phr_lab_avg.append([" ".join(words_pa), " ".join(words_la), avg])
    
    return str(phr_lab_avg)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)

