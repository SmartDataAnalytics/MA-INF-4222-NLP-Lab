# -*- coding: utf-8 -*-

import re

def initializeEnSentence(FREQDICT, sentence):
    words = sentence.strip().split()
    taggedSen = []
    for word in words:
        if word in ["“", "”", "\""]:
            taggedSen.append("''/" + FREQDICT["''"])
            continue
        
        tag = ''
        lowerW = word.lower()
        if word in FREQDICT:
            tag = FREQDICT[word] 
        elif lowerW in FREQDICT:
            tag = FREQDICT[lowerW] 
        else:
            if (re.search(r"([0-9]+-)|(-[0-9]+)", word) != None):
                tag = "JJ"
            elif (re.search(r"[0-9]+", word) != None):
                tag = "CD"
            elif (re.search(r'(.*ness$)|(.*ment$)|(.*ship$)|(^[Ee]x-.*)|(^[Ss]elf-.*)', word) != None):
                tag = "NN"
            elif (re.search(r'.*s$', word) != None and word[0].islower()):
                tag = "NNS"
            elif (word[0].isupper()):
                tag = "NNP"
            elif(re.search(r'(^[Ii]nter.*)|(^[nN]on.*)|(^[Dd]is.*)|(^[Aa]nti.*)', word) != None):
                tag = "JJ"
            elif (re.search(r'.*ing$', word) != None and word.find("-") < 0):
                tag = "VBG"
            elif (re.search(r'.*ed$', word) != None and word.find("-") < 0):
                tag = "VBN"
            elif (re.search(r'(.*ful$)|(.*ous$)|(.*ble$)|(.*ic$)|(.*ive$)|(.*est$)|(.*able$)|(.*al$)', word) != None
                  or word.find("-") > -1):
                tag = "JJ"
            elif(re.search(r'.*ly$', word) != None):
                tag = "RB"
            else:
                tag = "NN" 
                    
        taggedSen.append(word + "/" + tag)
                                           
    return " ".join(taggedSen)

def initializeEnCorpus(FREQDICT, inputFile, outputFile):
    lines = open(inputFile, "r").readlines()
    fileOut = open(outputFile, "w")
    for line in lines:
        fileOut.write(initializeEnSentence(FREQDICT, line) + "\n")
    fileOut.close()
