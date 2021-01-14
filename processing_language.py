import numpy as np
import pandas as pd
import nltk
stopwords = nltk.corpus.stopwords.words('english')

def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None

def abstract_to_feature(text):
    ### tokenizing
    tokens = nltk.word_tokenize(text)

    ### stopword removal
    filtered = [i for i in tokens if i not in stopwords]

    ### part-of-speech tagging
    tagged = nltk.pos_tag(filtered)
    return tagged

    ### lemmatizing
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed = []
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged))  # simplify PoS tags to lemmatize
    for simple, full in zip(wordnet_tagged, tagged):
        if simple[1] is None:
            processed.append(full)
        else:
            tmp = lemmatizer.lemmatize(simple[0], simple[1])
            processed.append((tmp, full[1]))
    
    return processed