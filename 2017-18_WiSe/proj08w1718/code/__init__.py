from labprojecttf.trainCNN import CNN_model
from labprojecttf.constructsentimentmaps import ConstructSentimentMaps
import numpy as np
import tensorflow as tf
import tflearn

review_pos = """This film has a special place in my heart, as when I caught it the first time, I was teaching adult literacy. 
It rang very true to me and even an outstanding student I had at the time. 
There are scenes which make you gulp with sudden emotion, and those which even put a smile on your face through sheer identification with the characters and their situation. 
Excellent performances by Jane Fonda and Robert DeNiro that rank with their best work, a great turn by a young Martha Plimpton, 
an inspiring story line, and a haunting musical score makes for a most enjoyable and rewarding experience."""

review_neg ="""Rita Hayworth plays a Brooklyn nightclub dancer named Rusty who specializes in cheesecake chorus revues; she manages to get herself on the cover of a national fashion magazine, but her impending success as a solo (with romantic offers all around) has smitten boss Gene Kelly chomping at the bit. 
Terribly tired piece of Technicolor cotton candy, with unmemorable musical sketches (the two worst of which are irrelevant flashbacks to the 1890s, with Hayworth portraying her own grandmother). 
Kelly, as always, dances well but acts with false sincerity; when he's serious, he's insufferable, and the rest of the time he's flying on adrenaline. 
The script is a lead weight, not even giving supporting players Phil Silvers and Eve Arden any good lines. *1/2 from ****"""


def GetReviewPolarity(review):
    sentiment_map = ConstructSentimentMaps(review)
    map = np.dstack([sentiment_map['pos'], sentiment_map['neg'], sentiment_map['so']]).reshape([1, 10, 10, 3])
    prediction = CNN_model.predict_label(map)
    return prediction[0,0]

print(GetReviewPolarity(review_pos))
print(GetReviewPolarity(review_neg))