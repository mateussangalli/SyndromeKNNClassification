import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import argparse

DATA_PATH = 'mini_gm_public_v0.1.p'

def get_class_distribution(data):
    """
    Traverse the data and save the quantity of images / subjects in each class as a csv file.
    """
    data_quantitities = {'syndrome_id': list(), 'num_images': list(), 'num_subjects': list()}

    label_names = list()
    for syndrome_index, (k_syn, v_syn) in enumerate(data.items()):
        num_images = 0
        num_subjects = len(v_syn)
        label_names.append(k_syn)
        for k_sub, v_sub in v_syn.items():
            num_images += len(v_sub)

        data_quantitities['syndrome_id'].append(k_syn)
        data_quantitities['num_images'].append(num_images)
        data_quantitities['num_subjects'].append(num_subjects)

    data_quantitities = pd.DataFrame(data=data_quantitities)
    data_quantitities.to_csv('data_quantities.csv')


def plot_tsne_graphs(data):
    """
    Plot TSNE graphs with multiple perplexity values.
    """

    vectors = list()
    y = list()
    label_names = list()
    for syndrome_index, (k_syn, v_syn) in enumerate(data.items()):
        label_names.append(k_syn)
        for k_sub, v_sub in v_syn.items():
            for k_im, v_im in v_sub.items():
                vectors.append(v_im)
                y.append(syndrome_index)

    vectors = np.stack(vectors, 0)
    y = np.stack(y, 0)
    perplexity_values = [6., 12., 24., 30., 50.]

    for i, p in enumerate(perplexity_values):
        tsne = TSNE(perplexity=p)
        reduced_vectors = tsne.fit_transform(vectors)

        plt.subplot(1, len(perplexity_values), i+1)
        for i in range(10):
            plt.scatter(reduced_vectors[y==i, 0], reduced_vectors[y==i, 1], label=label_names[i], cmap='tab10')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.savefig(f"tsne.pdf", bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)

    get_class_distribution(data)
    plot_tsne_graphs(data)

