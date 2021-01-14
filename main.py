from csv import reader
import numpy as np
import pandas as pd
from get_abstracts import get_abstracts
from processing_language import abstract_to_feature

###########################
###### query pubmed #######
###########################
keyword = 'peptide'
get_abstracts(keyword, 'steff.taelman@ugent.be')

###########################
### feature engineering ###
###########################
#handle = 'data/' + keyword + '_abstracts.csv'
#main_df = pd.read_csv(handle, index_col=0) #if memory issues occur, play with chuncksizes and iterators
#main_df = main_df[main_df['abstract'].notna()] #check how many these are (if not a lot, add manually)
#PoS = {} # NLP preprocessing
#for i in main_df.index:
#    PoS[i] = abstract_to_feature(main_df.iloc[i].abstract)

#features = [] # TF-IDF vectorisation