from csv import reader
import nltk
stopwords = nltk.corpus.stopwords.words('english')
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from querying import get_abstracts
from processing_language import abstract_to_BagofWords, identity_tokenizer, best_no_of_topics, topic_table
from visualize import umap_topic_vis, plot_residuals

SAMPLE_SIZE = 1000000


### query pubmed
keyword = 'peptide'
get_abstracts(keyword, 'steff.taelman@ugent.be', n=SAMPLE_SIZE, chuncksize=25000)




### feature engineering
#keyword = 'peptide_paper'
handle = 'data/' + keyword + '_abstracts.csv'
main_df = pd.read_csv(handle)                               #if memory issues occur, play with chuncksizes and iterators
main_df = main_df[main_df['abstract'].notna()]              #check how many these are (if not a lot, add manually)
print('Processing text...')                                 # NLP preprocessing
BoW_ordered = [] 
for i in main_df.index[:SAMPLE_SIZE]:
    tmp = abstract_to_BagofWords(main_df.loc[i].abstract, stopwords=stopwords)
    BoW_ordered.append(tmp)

print('Vectorizing...')                                     # TF-IDF vectorisation
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, max_df=0.95, min_df=3)    
features = tfidf.fit_transform(BoW_ordered)
print(features.shape)

#print('Determining a coherent set of topics...')            #use coherence score later to find optimal number of components
n = best_no_of_topics(BoW_ordered, range=(5,25), step=1, visualize=True)
#n = 13



### Clustering
print('Clustering abstracts into {} topics...'.format(n))
nmf = NMF(n_components=n, init='nndsvd').fit(features) 
n_top_words = 10
topic_df = topic_table(nmf, tfidf, n_top_words).T
print(topic_df.T)

docweights = nmf.transform(tfidf.transform(BoW_ordered))    # Getting a df with each topic by document
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

print('Checking residuals...')
df_topics = plot_residuals(BoW_ordered, nmf, tfidf, df_topics, sample_size=SAMPLE_SIZE, save_df=True)


print('Visualizing...')                                    # visualizing using UMAP and validating with some key papers
umap_topic_vis(nmf, features, topic_list=list(map(int, df_topics.topic_num)), cutoff=True)
#saap_abstract = """Development of novel antimicrobial agents is a top priority in the fight against multidrug-resistant (MDR) and persistent bacteria. We developed a panel of synthetic antimicrobial and antibiofilm peptides (SAAPs) with enhanced antimicrobial activities compared to the parent peptide, human antimicrobial peptide LL-37. Our lead peptide SAAP-148 was more efficient in killing bacteria under physiological conditions in vitro than many known preclinical- and clinical-phase antimicrobial peptides. SAAP-148 killed MDR pathogens without inducing resistance, prevented biofilm formation, and eliminated established biofilms and persister cells. A single 4-hour treatment with hypromellose ointment containing SAAP-148 completely eradicated acute and established, biofilm-associated infections with methicillin-resistant Staphylococcus aureus and MDR Acinetobacter baumannii from wounded ex vivo human skin and murine skin in vivo. Together, these data demonstrate that SAAP-148 is a promising drug candidate in the battle against antibiotic-resistant bacteria that pose a great threat to human health."""
#ll37_abstract = """Burn wound infections are often difficult to treat due to the presence of multidrug-resistant bacterial strains and biofilms. Currently, mupirocin is used to eradicate methicillin-resistant Staphylococcus aureus (MRSA) from colonized persons; however, mupirocin resistance is also emerging. Since we consider antimicrobial peptides to be promising candidates for the development of novel anti-infective agents, we studied the antibacterial activities of a set of synthetic peptides against different strains of S. aureus, including mupirocin-resistant MRSA strains. The peptides were derived from P60.4Ac, a peptide based on the human cathelicidin LL-37. The results showed that peptide 10 (P10) was the only peptide more efficient than P60.4Ac, which is better than LL-37, in killing MRSA strain LUH14616. All three peptides displayed good antibiofilm activities. However, both P10 and P60.4Ac were more efficient than LL-37 in eliminating biofilm-associated bacteria. No toxic effects of these three peptides on human epidermal models were detected, as observed morphologically and by staining for mitochondrial activity. In addition, P60.4Ac and P10, but not LL-37, eradicated MRSA LUH14616 and the mupirocin-resistant MRSA strain LUH15051 from thermally wounded human skin equivalents (HSE). Interestingly, P60.4Ac and P10, but not mupirocin, eradicated LUH15051 from the HSEs. None of the peptides affected the excretion of interleukin 8 (IL-8) by thermally wounded HSEs upon MRSA exposure. In conclusion, the synthetic peptides P60.4Ac and P10 appear to be attractive candidates for the development of novel local therapies to treat patients with burn wounds infected with multidrug-resistant bacteria."""
#p10_abstract = """Skin bacterial colonization/infection is a frequent cause of morbidity in patients with chronic wounds and allergic/inflammatory skin diseases. This study aimed to develop a novel approach to eradicate meticillin-resistant Staphylococcus aureus (MRSA) from human skin. To achieve this, the stability and antibacterial activity of the novel LL-37-derived peptide P10 in four ointments was compared. Results indicate that P10 is chemically stable and antibacterial in hypromellose gel and Softisan-containing cream, but not in Cetomacrogol cream (with or without Vaseline), at 4 °C for 16 months. Reduction in MRSA counts on Leiden human epidermal models (LEMs) by P10 in hypromellose gel was greater than that of the peptide in Cetomacrogol cream or phosphate buffered saline. P10 did not show adverse effects on LEMs irrespective of the ointment used, while Cetomacrogol with Vaseline and Softisan cream, but not hypromellose gel or Cetomacrogol cream, destroyed MRSA-colonized LEMs. Taking all this into account, P10 in hypromellose gel dose-dependently reduced MRSA colonizing the stratum corneum of the epidermis as well as biofilms of this bacterial strain on LEMs. Moreover, P10 dose-dependently reduced MRSA counts on ex-vivo human skin, with P10 in hypromellose gel being more effective than P10 in Cetomacrogol and Softisan creams. P10 in hypromellose gel is a strong candidate for eradication of MRSA from human skin."""
#op145_abstract = """Emergence of multidrug-resistant (MDR) bacteria is becoming a major health concern worldwide as patients infected with such strains can no longer be successfully treated. A promising alternative treatment strategy is the use of antimicrobial peptides (AMPs). Within the EU-funded BALI consortium, we aimed to incorporate immune orchestrating synthetic antimicrobial and anti-biofilm peptides into a novel controlled release drug delivery formulation to control biofilm infections. The basis for the development of the AMPs was OP-145, previously named P60.4Ac, a 24 amino acid peptide derived from the human cathelicidin LL-37. OP-145 killed Staphylococcus aureus in vitro and eradicated this species from three-dimensional wounded human skin models. Moreover, OP-145 inhibited the formation of biofilms by S. aureus on abiotic and biotic surfaces. To get more insight into the mechanism of antimicrobial and anti-biofilm activity of this peptide, we performed proteomic and phenotype microarray analyses on S. aureus exposed to sub-inhibitory concentrations of OP-145. The knowledge obtained using this approach is vital to guide the design of novel and improved antimicrobial and anti-biofilm peptides."""
#antitumor_antiviral_abstract = """Cationic antimicrobial peptides (AMPs) and host defense peptides (HDPs) show vast potential as peptide-based drugs. Great effort has been made in order to exploit their mechanisms of action, aiming to identify their targets as well as to enhance their activity and bioavailability. In this review, we will focus on both naturally occurring and designed antiviral and antitumor cationic peptides, including those here called promiscuous, in which multiple targets are associated with a single peptide structure. Emphasis will be given to their biochemical features, selectivity against extra targets, and molecular mechanisms. Peptides which possess antitumor activity against different cancer cell lines will be discussed, as well as peptides which inhibit virus replication, focusing on their applications for human health, animal health and agriculture, and their potential as new therapeutic drugs. Moreover, the current scenario for production and the use of nanotechnology as delivery tool for both classes of cationic peptides, as well as the perspectives on improving them is considered."""

print('Done!')