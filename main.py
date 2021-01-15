from csv import reader
import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from get_abstracts import get_abstracts
from processing_language import abstract_to_BagofWords, identity_tokenizer, best_no_of_topics, topic_table

### query pubmed
keyword = 'peptide'
#get_abstracts(keyword, 'steff.taelman@lizard.bio', n=10000)




### feature engineering
#keyword = 'peptide_paper'
handle = 'data/' + keyword + '_abstracts.csv'
main_df = pd.read_csv(handle, index_col=0)                  #if memory issues occur, play with chuncksizes and iterators
main_df = main_df[main_df['abstract'].notna()]              #check how many these are (if not a lot, add manually)
print('Processing text...')                                 # NLP preprocessing
BoW_ordered = [] 
for i in main_df.index[:1000]:
    tmp = abstract_to_BagofWords(main_df.loc[i].abstract)
    BoW_ordered.append(tmp)

print('Vectorizing...')                                     # TF-IDF vectorisation
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)    
features = tfidf.fit_transform(BoW_ordered)

print('Determining a coherent set of topics...')            #use coherence score later to find optimal number of components
n = best_no_of_topics(BoW_ordered, range=(5,25), step=1, visualize=True)




### Clustering
print('Clustering abstracts into {} topics...'.format(n))
nmf = NMF(n_components=n, init='nndsvd').fit(features) 
topic_df = topic_table(nmf, tfidf, n_top_words).T
print(topic_df.T.head(n=8))

docweights = nmf.transform(tfidf.transform(BoW_ordered))    # Getting a df with each topic by document
n_top_words = 8
topic_df = topic_table(nmf, tfidf, n_top_words).T
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1)
topic_df['topics'] = topic_df['topics'].str[0]

topic_df = topic_df['topics'].reset_index()                 # Merge topics into main dataframe
topic_df.columns = ['topic_num', 'topics']
pmid = main_df['pubmed_id'].tolist()[:SAMPLE_SIZE]
df_temp = pd.DataFrame({'pubmed_id': pmid, 'topic_num': map(int, docweights.argmax(axis=1))})
merged_topic = df_temp.merge(topic_df, on='topic_num', how='left')
df_topics = pd.merge(main_df, merged_topic, on='pubmed_id', how='left')
df_topics = df_topics.drop('abstract', axis=1)

#print('Visualizing...')                                    # visualizing using UMAP and validating with some key papers

print('Done!')