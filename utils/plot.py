import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne(data,case,path,config,PN=False):
    loss = config['loss']
    policy = config['policy']
    data = np.asarray(data)
    case = np.asarray(case)
    unique = np.unique(case)
    filtered_case = np.zeros_like(case,dtype=np.int32)
    filtered_case = np.where(case!=1.,filtered_case,1)
    filtered_case = np.where(case!=3.,filtered_case,1)
    num_cases = unique.shape[0]
    if data.shape[1]>2:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data)
    else:
        tsne_results = data
    plt.figure(figsize=(7,9))
    plt.title("TSNE latent {} {}".format(loss,policy))
    if PN:
        sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=filtered_case,
            palette=sns.color_palette("hls",2),
            legend="full",
            alpha=0.4
        )
        plt.savefig(path+'test_res_2.png')
    else:
        sns.scatterplot(
            x=tsne_results[:,0], y=tsne_results[:,1],
            hue=case,
            palette=sns.color_palette("viridis",num_cases),
            legend="full",
            alpha=0.4
        )
        plt.legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.savefig(path+'test_res.png')