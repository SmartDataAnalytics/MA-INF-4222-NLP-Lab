#!/usr/bin/python
#
# This example shows how to use the MITIE Python API to perform named entity
# recognition and also how to run a binary relation detector on top of the
# named entity recognition outputs.
#
import sys, os
# Make sure you put the mitielib folder into the python search path.  There are
# a lot of ways to do this, here we do it programmatically with the following
# two statements:
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent + '/../../mitielib')

from mitie import *
from collections import defaultdict
import numpy as np




print("loading NER model...")
#ner_w_model = named_entity_extractor('../../MITIE-models/english/new_ner_model.dat')
ner_model = named_entity_extractor('../../MITIE-models/english/ner_model.dat')
#print("\nTags output by this NER model:", ner.get_possible_ner_tags())

# Load a text file and convert it into a list of words.

tokens = []
with open('../../emerging.test.conll') as file:
		for line in file:
			s = (line.rstrip('\n')).split('\t')
			if s[0]!='':
				tokens.append(s[0])

#tokens = tokenize(load_entire_file('../../sample_text.txt'))
#print("Tokenized input:", tokens)
#model_w_entities = ner_w_model.extract_entities(tokens)
mitie_entities = ner_model.extract_entities(tokens)

#print("\nEntities found:", entities[0])
print("\nNumber of entities detected by mitie model:", len(mitie_entities))
#print("\nNumber of entities detected by ner w model:", len(model_w_entities))



#for e in entities:
	#with open('output.txt', 'w') as the_file:
	#	the_file.write(e[0]+'\t'+e[1])
#	entity_text = " ".join(tokens[i].decode() for i in range)
#	print(entity_text+'\t'+e[1])
# entities is a list of tuples, each containing an xrange that indicates which
# tokens are part of the entity, the entity tag, and an associate score.  The
# entities are also listed in the order they appear in the input text file.
# Here we just print the score, tag, and text for each entity to the screen.
# The larger the score the more confident MITIE is in its prediction.
'''
with open('w_output.txt', 'w') as the_file:
	for e in model_w_entities:
		range = e[0]
		tag = e[1]
		score = e[2]
		score_text = "{:0.3f}".format(score)
		entity_text = " ".join(tokens[i] for i in range)   
		the_file.write(entity_text+"\t"+tag+"\n")
		#print("   Score: " + score_text + ": " + tag + ": " + entity_text)
		#print("   Score: " + score_text + ": " + tag )
'''
d = []
with open('mitie_output.txt', 'w') as the_file:
	for e in mitie_entities:
		range = e[0]
		tag = e[1]
		score = e[2]
		score_text = "{:0.3f}".format(score)
		entity_text = " ".join(tokens[i] for i in range) 
		#print(len(entity_text.split()))
		if tag == 'ORGANIZATION':
			tag = 'corporation'
		if(len(entity_text.split())>1):
#			print(entity_text)
			for i, val in enumerate(entity_text.split(), 0):
				if(i==0):				
					the_file.write(val+"\t"+('B-'+tag)+"\n")
				else:
					the_file.write(val+"\t"+('I-'+tag)+"\n")				
		else:			
			the_file.write(entity_text+"\t"+('B-'+tag)+"\n")
		d.append(entity_text.split())
toks = []
l = [x for x in tokens if x not in d]
for item in tokens:
	if item in d:
		print(item)		
		pass	
	else:	
		toks.append(item)
		

with open('m.txt', 'w') as the_file:
	the_file.write("\n".join(toks))		

print(len(tokens),len(d),len(l))

'''


# Now let's run one of MITIE's binary relation detectors.  MITIE comes with a
# bunch of different types of relation detector and includes tools allowing you
# to train new detectors.  However, here we simply use one, the "person born in
# place" relation detector.
rel_detector = binary_relation_detector("../../MITIE-models/english/binary_relations/rel_classifier_people.person.place_of_birth.svm")

# First, let's make a list of neighboring entities.  Once we have this list we
# will ask the relation detector if any of these entity pairs is an example of
# the "person born in place" relation.
neighboring_entities = [(entities[i][0], entities[i+1][0]) for i in xrange(len(entities)-1)]
# Also swap the entities and add those in as well.  We do this because "person
# born in place" mentions can appear in the text in as "place is birthplace of
# person".  So we must consider both possible orderings of the arguments.
neighboring_entities += [(r,l) for (l,r) in neighboring_entities]

# Now that we have our list, let's check each entity pair and see which one the
# detector selects.
for person, place in neighboring_entities:
    # Detection has two steps in MITIE. First, you convert a pair of entities
    # into a special representation.
    rel = ner.extract_binary_relation(tokens, person, place)
    # Then you ask the detector to classify that pair of entities.  If the
    # score value is > 0 then it is saying that it has found a relation.  The
    # larger the score the more confident it is.  Finally, the reason we do
    # detection in two parts is so you can reuse the intermediate rel in many
    # calls to different relation detectors without needing to redo the
    # processing done in extract_binary_relation().
    score = rel_detector(rel)
    # Print out any matching relations.
    if (score > 0):
        person_text     = " ".join(tokens[i].decode() for i in person)
        birthplace_text = " ".join(tokens[i].decode() for i in place)
        print(person_text, "BORN_IN", birthplace_text)

# The code above shows the basic details of MITIE's relation detection API.
# However, it is important to note that real world data is noisy any confusing.
# Not all detected relations will be correct.  Therefore, it's important to
# aggregate many relation detections together to get the best signal out of
# your data.  A good way to do this is to pick an entity you are in interested
# in (e.g. Benjamin Franklin) and then find all the relations that mention him
# and order them by most frequent to least frequent.  We show how to do this in
# the code below.
query = "Benjamin Franklin"
hits = defaultdict(int)

for person, place in neighboring_entities:
    rel = ner.extract_binary_relation(tokens, person, place)
    score = rel_detector(rel)
    if (score > 0):
        person_text     = " ".join(tokens[i].decode() for i in person)
        birthplace_text = " ".join(tokens[i].decode() for i in place)
        if (person_text == query):
            hits[birthplace_text] += 1

print("\nTop most common relations:")
for place, count in sorted(hits.items(), key=lambda x:x[1], reverse=True):
    print(count, "relations claiming", query, "was born in", place)


'''
