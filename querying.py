import numpy as np
import pandas as pd
from pymed import PubMed
from tqdm import tqdm

def get_abstracts(query, email, n=10000, chuncksize=50000):
    pubmed = PubMed(tool="PubMedSearcher", email=email)

    search_term = query
    print('Querying PubMed...')
    results = pubmed.query(search_term, max_results=n)
    articleList = []
    print('Gathering articles...')
    for article in tqdm(results):
        # Print the type of object we've found (can be either PubMedBookArticle or PubMedArticle).
        # We need to convert it to dictionary with available function
        articleDict = article.toDict()
        articleList.append(articleDict)

    print('Compiling in chuncks...')
    filename ='data/' + query + '_abstracts.csv'
    pre = 0
    for chunck in tqdm(range(chuncksize, n, chuncksize)): #write out in chuncksizes of 50 000
        articleInfo = []
        for article in articleList[pre:chunck]:
            #Sometimes article['pubmed_id'] contains list separated with comma - take first pubmedId in that list - thats article pubmedId
            pubmedId = article['pubmed_id'].partition('\n')[0]
            # Append article info to dictionary 
            articleInfo.append({u'pubmed_id':pubmedId,
                                u'title':article['title'],
                                u'abstract':article['abstract'],
                                u'doi':article['doi']})
        # Specify when to write or append to existing file
        if pre == 0: 
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False
            
        # Generate Pandas DataFrame from list of dictionaries
        articles_df = pd.DataFrame.from_dict(articleInfo)
        export_csv = articles_df.to_csv(filename, mode=mode, index = None, header=header) 
        pre = chunck

    # gather last chunck
    articleInfo = []
    for article in articleList[chunck:]:
        pubmedId = article['pubmed_id'].partition('\n')[0]
        articleInfo.append({u'pubmed_id':pubmedId,
                            u'title':article['title'],
                            u'abstract':article['abstract'],
                            u'doi':article['doi']})
    articles_df = pd.DataFrame.from_dict(articleInfo)
    export_csv = articles_df.to_csv(filename, mode='a', index = None, header=False) 

    print('Exported abstracts to {}!'.format(filename))

