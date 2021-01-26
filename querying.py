from bs4 import *
import numpy as np
import pandas as pd
from pymed import PubMed
import re
from tqdm import tqdm
import urllib

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




def query_mapping(pmids, email, conversion_dict):
    assert len(pmids) <= 200
    ids = ','.join(map(str, pmids))
    url = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids=' + ids + '&email=' + email
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html)
    for i in soup.find_all('record'):
        tmp = re.search(r'pmcid="PMC\d+', str(i))
        if tmp:
            pm = int(re.search(r'requested-id="\d+', str(i)).group()[14:])
            pmcid = tmp.group()[7:]
            conversion_dict[pm] = pmcid
    return conversion_dict




def pmids_to_file(pmids, conversion_dict, emails):
    unmapped = list(set(pmids).difference(set(conversion_dict.keys())))
    chuncks = [unmapped[i:i + 200] for i in range(0, len(unmapped), 200)]
    for chunck, email in zip(chuncks, emails):
        conversion_dict = query_mapping(chunck, email, conversion_dict)
    
    files = []
    for i in pmids:
        if i in conversion_dict:
            handle = conversion_dict[i] + '.txt'
            files.append(handle)
    return files