import numpy as np
from sklearn.model_selection import StratifiedKFold
from knn import KNNClassifier, euclidean_metric, cosine_metric
from sklearn import metrics as skl_metrics
import os
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd


NUM_NEIGHBORS = 15
CROSS_VALIDATION_SPLITS = 10

data = np.load("mini_gm_public_v0.1.p", allow_pickle=True)


def divide_by_syndrome(data):
    subjects = list()
    subject_dict = dict()
    y = list()
    subject_index = 0
    for syndrome_index, (k_syn, v_syn) in enumerate(data.items()):
        for k_sub, v_sub in v_syn.items():
            subject_dict[subject_index] = k_sub
            subject_index += 1
            subjects.append(v_sub)
            y.append(syndrome_index)

    y = np.array(y)
    return subjects, y, subject_dict


def get_images_from_subjects(subjects, subject_labels):
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
    fpr = list()
    tpr = list()
    for t in thresholds:
        predictions = probabilities > t
        false_positives = np.logical_and(np.logical_not(true_labels), predictions)
        fpr.append(false_positives.sum() / (np.logical_not(predictions).sum() + 1e-5))

        true_positives = np.logical_and(true_labels, predictions)
        tpr.append(true_positives.sum() / (predictions.sum() + 1e-5))

    return np.array(fpr), np.array(tpr)


def evaluate(data, num_neighbors, metric_function, seed=None):

    subjects, labels, subject_dict = divide_by_syndrome(data)

    kfold = StratifiedKFold(n_splits=CROSS_VALIDATION_SPLITS, shuffle=True, random_state=seed)

    output_metrics = {'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        output_metrics[f'top{k}_accuracy'] = list()

    fpr_list = {i: list() for i in range(10)}
    tpr_list = {i: list() for i in range(10)}
    thresholds = np.linspace(0., 1., num_neighbors)

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

        for i in range(10):
            fpr, tpr = roc_curve(test_labels==i, probabilities[:, i], thresholds)
            fpr_list[i].append(fpr)
            tpr_list[i].append(tpr)

    fpr = {k: np.mean(np.stack(v, 0), axis=0) for k, v in fpr_list.items()}
    tpr = {k: np.mean(np.stack(v, 0), axis=0) for k, v in tpr_list.items()}
    # for i in range(10):
    #     plt.plot(fpr[i], tpr[i])
    #     plt.show()

    output_metrics = pd.DataFrame(output_metrics)

    return output_metrics


if __name__ == "__main__":
    seed = 999

    averages = {'name': list(), 'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        averages[f'top{k}_accuracy'] = list()

    stds = {'AUC': list(), 'precision': list(), 'recall': list(), 'fscore': list()}
    for k in range(1, 6):
        stds[f'top{k}_accuracy'] = list()

    for k in [5, 10, 15, 20]:
        metrics_cosine = evaluate(data, k, cosine_metric, seed)
        metrics_cosine.to_csv(os.path.join('results', f'metrics_cosine_{k}.csv'))

        metrics_euclidean = evaluate(data, k, euclidean_metric, seed)
        metrics_euclidean.to_csv(os.path.join('results', f'metrics_euclidean_{k}.csv'))

        print(f'cosine metric, {k=}')
        print(metrics_cosine.describe())

        print(f'euclidean metric, {k=}')
        print(metrics_euclidean.describe())

        averages['name'].append(f'cosine_{k}')
        averages['name'].append(f'euclidean_{k}')
        for col in metrics_cosine.columns:
            averages[col].append(metrics_cosine[col].mean())
            averages[col].append(metrics_euclidean[col].mean())
            stds[col].append(metrics_cosine[col].std())
            stds[col].append(metrics_euclidean[col].std())

    for k, v in stds.items():
        averages[f'{k}_std'] = v

    df = pd.DataFrame(averages)
    df.to_csv(os.path.join('results', 'average_results.csv'))



