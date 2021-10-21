import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne(data,case,path,config):
    loss = config['loss']
    policy = config['policy']
    data = np.asarray(data)
    case = np.asarray(case)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    plt.figure()
    # plt.subplot(2,1,1)
    # plt.title("test MSE LOSS")
    # plt.plot(fl,label='negative')
    # plt.plot(el,label='expert')

    # plt.subplot(2,1,2)
    plt.title("TSNE latent {} {}".format(loss,policy))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=case,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.3
    )
    plt.savefig(path+'test_res.png')