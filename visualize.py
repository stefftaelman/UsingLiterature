import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import umap     

def plot_residuals(tokenized_texts, nmf_model, vectorizer, df, sample_size=10000, save_df=True):
    """
    """
    # Save residuals for each document
    A = vectorizer.transform(tokenized_texts)                            
    W = nmf_model.components_
    H = nmf_model.transform(A)
    r = np.zeros(A.shape[0])
    for row in range(A.shape[0]):
        r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), 'fro')
    sum_sqrt_res = round(sum(np.sqrt(r)), 3)
    df.drop([i for i in range(sample_size, len(df))], inplace=True)
    df['resid'] = r
    resid_data = df[['topic_num', 'resid']].groupby('topic_num').mean().sort_values(by='resid')

    # plot residuals for each topic
    fig = plt.figure(figsize=(20,7))                            
    x = resid_data.index
    y = resid_data['resid']
    g = sb.barplot(x=x, y=y, order=x, palette='rocket')
    g.set_xticklabels(g.get_xticklabels(), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Topic Number', fontsize=14)
    plt.ylabel('Residual', fontsize=14)
    plt.title('Avg. Residuals by Topic Number', fontsize=17)
    fig.savefig('../deliverables/topic_residuals.png', dpi=fig.dpi, bbox_inches='tight')
    if save_df:
        return df

def umap_topic_vis(nmf_model, features, topic_list, cutoff=True, validation=True):
    """
    """
    palette = ['#a9a9a9', '#2f4f4f', '#556b2f', '#a52a2a', '#483d8b', 
               '#3cb371', '#000080', '#ffff00', '#00ff00', '#8a2be2',
               '#9acd32', '#8b008b', '#ff4500', '#ffa500', '#00ff7f', 
               '#00ffff', '#00bfff', '#0000ff', '#ff00ff', '#1e90ff', 
               '#db7093', '#eee8aa', '#ff1493', '#ffa07a', '#ee82ee']
    if cutoff:
        loop_values = [0.1, 0.15, 0.2, 0.25]
    else:
        loop_values = [0]

    for idx, i in enumerate(loop_values):
        useful_words = [idx for idx, i in enumerate(np.any(nmf_model.components_ > i, axis=0)) if i]
        umap_features = features[:, useful_words]

        fig = plt.figure(figsize=(10,10))
        ### visualisation
        reducer = umap.UMAP(init='spectral', n_neighbors=8, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(umap_features)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=[palette[x] for x in topic_list], s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the topics', fontsize=24)
        plt.legend(labels=list(set(topic_list)))
        filehandle = '../deliverables/UMAP_{}.png'.format(idx)
        fig.savefig(filehandle, dpi=fig.dpi, bbox_inches='tight')
        plt.show()
