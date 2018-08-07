import numpy as np
from sklearn.metrics import roc_auc_score
import pdb

def accuracy(y_pred, y_true, thresh=0.5, reverse=False):
    y = (y_pred >= thresh) if not reverse else (y_pred <= thresh)
    return np.mean(y == y_true)

def auc(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)

def find_clf_threshold(y_pred, reverse=False):
    thresh = 0.0
    accuracy = 0.0
    right = 0.0
    wrong = 0.0
    score_sorted = sorted(y_pred)
    for i in range(len(score_sorted)):
        t = score_sorted[i]
        for j in range(int(len(score_sorted)/2)):
            if (y_pred[j]>=t):
                right += 1
            else:
                wrong += 1
        for k in range(int(len(score_sorted)/2), int(len(score_sorted)),1):
            if (y_pred[k]<t):
                right += 1
            else:
                wrong += 1
        if (right/(len(score_sorted))>accuracy):
            accuracy = right/len(score_sorted)
            thresh = t
        right = 0.0
        wrong = 0.0
    return accuracy, thresh

def get_accuracy(y_pred, thresh):
    accuracy = 0.0
    right = 0.0
    wrong = 0.0
    score_sorted = sorted(y_pred)
    for j in range(int(len(score_sorted)/2)):
        if (y_pred[j]>=thresh):
            right += 1
        else:
            wrong += 1
    for k in range(int(len(score_sorted)/2), int(len(score_sorted)),1):
        if (y_pred[k]<thresh):
            right += 1
        else:
            wrong += 1
    return right/(len(score_sorted))

def stats(values):
    return '{0:.4f} +/- {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))


def evaluate_model(sub_, obj_, model, maxlen_s, maxlen_o, word_id_map, rel_id_map, eval_type):
    del_rels = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
    'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
    'NotHasA','NotIsA','NotHasProperty','NotCapableOf']

    for del_rel in del_rels:
        if del_rel.lower() in rel_id_map:
            del rel_id_map[del_rel.lower()]    
    id_rel_map = {v:k for k,v in rel_id_map.items()}
    sub_ = sub_.strip().split('_')
    obj_ = obj_.strip().split('_')
    count_s = [word for word in sub_ if word in word_id_map]
    count_o = [word for word in obj_ if word in word_id_map]
    if len(count_s) == 0 and len(count_p) == 0:
        print('No words in Vocabulary')
    elif len(count_s) == 0:
        print('All words in subject out of Vocabulary')
    elif len(count_o) == 0:
        print('All words in object out of Vocabulary')
    else:
        print('Words in subject found in Vocabulary',count_s)
        print('Words in object found in Vocabulary',count_o)
        sub_ = np.array([word_id_map[word] if word in word_id_map else word_id_map['UUUNKKK'] for word in sub_])
        obj_ = np.array([word_id_map[word] if word in word_id_map else word_id_map['UUUNKKK'] for word in obj_])
        pred_ =np.array([rel for rel in rel_id_map.values()])
        sub_ = np.concatenate((sub_, np.zeros(maxlen_s-len(sub_))))
        obj_ = np.concatenate((obj_, np.zeros(maxlen_o-len(obj_))))
        sub_ = np.repeat(sub_.reshape(-1,len(sub_)), len(pred_), axis=0)
        obj_ = np.repeat(obj_.reshape(-1,len(obj_)), len(pred_), axis=0)
        score_ = model.forward(sub_, obj_, pred_)
        prob_ = model.predict_proba(score_)
        prob_ = prob_.reshape(-1, len(prob_))[0]
        if eval_type == 'topfive':
            sort_score = np.argsort(prob_)[::-1]
            sort_score = sort_score[:5]
            for id_ in sort_score:
                if id_rel_map[pred_[id_]]!= 'UUUNKKK':
                    print(id_rel_map[pred_[id_]], 'score: ', prob_[id_])
