import numpy as np
from sklearn.model_selection import StratifiedKFold
from knn import KNNClassifier, euclidean_metric, cosine_metric
from sklearn import metrics as skl_metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


CROSS_VALIDATION_SPLITS = 10
NUM_CLASSES = 10
DATA_PATH = "mini_gm_public_v0.1.p"



def extract_subjects(data):
    """
    Takes the data dict and maps it to a list of subjects and a vector of integer labels for each syndrome.
    Also store a syndrome dict to be able to retrieve its name later.
    """
    subjects = list()
    syndrome_dict = dict()
    y = list()
    for syndrome_index, (k_syn, v_syn) in enumerate(data.items()):
        syndrome_dict[syndrome_index] = k_syn
        for k_sub, v_sub in v_syn.items():
            subjects.append(v_sub)
            y.append(syndrome_index)

    y = np.array(y)
    return subjects, y, syndrome_dict


def get_images_from_subjects(subjects, subject_labels):
    """
    Given a list of subjects and a vector of labels in the format of the output of the function extract_subjects.
    This function returns a list of images with their respective labels, and an integer vector groups indicating from which subject it came from.
    """
    images = list()
    labels = list()
    groups = list()
    for i, (sub, lab) in enumerate(zip(subjects, subject_labels)):
        images += sub.values()
        labels += len(sub) * [lab]
        groups += len(sub) * [i]

    images = np.stack(images, 0)
    labels = np.array(labels)[:, np.newaxis]
    groups = np.stack(groups, 0)

    return images, labels, groups

def roc_curve(true_labels, probabilities, thresholds):
    """
    Returns the False Positive Rate (fpr) and True Positive Rate (tpr) necessary for plotting the ROC curve.
    Computes the multiclass ROC curve by macro-averaging, i.e. computes individual ROC curves for each class and then averages the results afterwards.

    Arguments:
    true_labels -- ground truth vector of integer labels.
    probabilities -- float matrix, probability of classifying each class.
    threshold -- thresholds for classification. Ideally it contains all unique probability values.
    """

    fpr = list()
    tpr = list()
    one_hot_labels = np.eye(NUM_CLASSES)[true_labels[:, 0]]

    for t in thresholds:
        predictions = np.where(probabilities > t, 1., 0.)
        false_positives = np.minimum(1-one_hot_labels, predictions).sum(axis=0)
        true_positives = np.minimum(one_hot_labels, predictions).sum(axis=0)


        true_cases = np.sum(one_hot_labels, axis=0)
        false_cases = one_hot_labels.shape[0] - true_cases

        fpr.append(false_positives / false_cases)
        tpr.append(true_positives / true_cases)

    return np.mean(np.stack(fpr, 0), 1), np.mean(np.stack(tpr, 0), 1)


def evaluate(data, num_neighbors, metric_function, seed=None):
    """
    Performs an evaluation of the KNN algorithm with given number of neighbors and metric function.
    Arguments:
    data -- input dictionnaire of the data.
    num_neighbors -- number of neighbors used for the KNN classifier.
    metric_function -- function that computes the distances between the training and test vectors.
    seed -- random seed used for the K-Fold splits to ensure reproducibility.

    Returns:
    output_metrics -- pandas DataFrame with the evaluation metrics for each of the splits.
    (fpr, tpr) -- tuple with the averages of the false and true positive rates for plotting the ROC curve.
    """
    subjects, labels, syndrome_dict = extract_subjects(data)

    kfold = StratifiedKFold(n_splits=CROSS_VALIDATION_SPLITS, shuffle=True, random_state=seed)

    output_metrics = {'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        output_metrics[f'top{k}_accuracy'] = list()

    fpr_list = list()
    tpr_list = list()
    thresholds = np.arange(-1, num_neighbors+1) / num_neighbors

    for gallery_indices, test_indices in kfold.split(subjects, labels):
        gallery_subjects = [subjects[i] for i in gallery_indices]
        gallery_labels = labels[gallery_indices]
        gallery_images, gallery_labels, _ = get_images_from_subjects(
            gallery_subjects, gallery_labels
        )

        test_subjects = [subjects[i] for i in test_indices]
        test_labels = labels[test_indices, np.newaxis]
        test_images, _, test_groups = get_images_from_subjects(test_subjects, test_labels)

        knn_classifier = KNNClassifier(num_neighbors, metric_function)
        knn_classifier.fit(gallery_images, gallery_labels)
        probabilities = knn_classifier.predict_proba(test_images, test_groups)

        for k in range(1, 6):
            output_metrics[f'top{k}_accuracy'].append(skl_metrics.top_k_accuracy_score(test_labels, probabilities, k=k))

        prediction = np.argmax(probabilities, -1)

        output_metrics['AUC'].append(skl_metrics.roc_auc_score(test_labels, probabilities, multi_class='ovr'))
        output_metrics['precision'].append(skl_metrics.precision_score(test_labels, prediction, average='macro'))
        output_metrics['recall'].append(skl_metrics.recall_score(test_labels, prediction, average='macro'))
        output_metrics['fscore'].append(skl_metrics.f1_score(test_labels, prediction, average='macro'))

        fpr, tpr = roc_curve(test_labels, probabilities, thresholds)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr = np.mean(np.stack(fpr_list, 0), axis=0)
    tpr = np.mean(np.stack(tpr_list, 0), axis=0)

    output_metrics = pd.DataFrame(output_metrics)

    return output_metrics, (fpr, tpr)

def plot_roc_curves(roc_cos, roc_euc, auc_cos, auc_euc, output_path):
    """
    Plot the two ROC curves
    """

    plt.figure()
    plt.plot((0,1), (0,1), color='red')

    fpr_cos, tpr_cos = roc_cos
    fpr_euc, tpr_euc = roc_euc

    plt.step(fpr_cos, tpr_cos, color='blue', label=f'cosine (AUC={auc_cos:.2f})', where='post')
    plt.step(fpr_euc, tpr_euc, color='green', label=f'Euclidean (AUC={auc_euc:.2f})', where='post')
    plt.grid()
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for both distances')
    plt.savefig(output_path, bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    args = parser.parse_args()

    seed = 999
    data = np.load(args.data_path, allow_pickle=True)

    # create a table to save the averages and standard deviations of the results
    averages = {'name': list(), 'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        averages[f'top{k}_accuracy'] = list()

    stds = {'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        stds[f'top{k}_accuracy'] = list()

    for k in [5, 10, 15, 20]:
        # compute the results and roc curve for the cosine distance
        metrics_cosine, roc_cos = evaluate(data, k, cosine_metric, seed)
        metrics_cosine.to_csv(os.path.join('results', f'metrics_cosine_{k}.csv'))

        # compute the results and roc curve for the euclidean distance
        metrics_euclidean, roc_euc = evaluate(data, k, euclidean_metric, seed)
        metrics_euclidean.to_csv(os.path.join('results', f'metrics_euclidean_{k}.csv'))

        # save roc curves
        plot_roc_curves(roc_cos, roc_euc, metrics_cosine['AUC'].mean(), metrics_euclidean['AUC'].mean(), os.path.join('results', f'roc_curves_{k}.pdf'))

        # print the statistics of this fold's results
        print(f'cosine metric, {k=}')
        print(metrics_cosine.describe())

        print(f'euclidean metric, {k=}')
        print(metrics_euclidean.describe())

        # add the results to the result table
        averages['name'].append(f'cosine_{k}')
        averages['name'].append(f'euclidean_{k}')
        for col in metrics_cosine.columns:
            averages[col].append(metrics_cosine[col].mean())
            averages[col].append(metrics_euclidean[col].mean())
            stds[col].append(metrics_cosine[col].std())
            stds[col].append(metrics_euclidean[col].std())

    # merge the averages and std tables into one
    for k, v in stds.items():
        averages[f'{k}_std'] = v

    df = pd.DataFrame(averages)
    df.to_csv(os.path.join('results', 'average_results.csv'))



