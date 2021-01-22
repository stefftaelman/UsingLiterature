import numpy as np
import pandas as pd
from pymed import PubMed
from tqdm import tqdm

def get_abstracts(query, email, n=10000):
    pubmed = PubMed(tool="PubMedSearcher", email=email)

    search_term = query
    print('Querying PubMed...')
    results = pubmed.query(search_term, max_results=n)
    articleList = []
    articleInfo = []
    print('Gathering articles...')
    for article in tqdm(results):
        # Print the type of object we've found (can be either PubMedBookArticle or PubMedArticle).
        # We need to convert it to dictionary with available function
        articleDict = article.toDict()
        articleList.append(articleDict)

    for article in articleList:
        #Sometimes article['pubmed_id'] contains list separated with comma - take first pubmedId in that list - thats article pubmedId
        pubmedId = article['pubmed_id'].partition('\n')[0]
        # Append article info to dictionary 
        articleInfo.append({u'pubmed_id':pubmedId,
                        u'title':article['title'],
                        u'abstract':article['abstract'],
                        u'doi':article['doi']})

    # Generate Pandas DataFrame from list of dictionaries
    print('Compiling...')
    articlesPD = pd.DataFrame.from_dict(articleInfo)
    filename ='data/' + query + '_abstracts.csv'
    export_csv = articlesPD.to_csv(filename, index = None, header=True) 
    print('Exported abstracts to {}!'.format(filename))

