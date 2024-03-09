import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Optional

EPS = 1e-5

class KNNClassifier:
    def __init__(self, k: int, metric_function: Callable[[ArrayLike, ArrayLike], NDArray]):
        """
        Initializes the classifier.

        Arguments:
            k -- number of neighbors used for classification.
            metric_function -- callable that takes two rank-2 numpy arrays, the gallery and test matrices, 
            and returns a matrix where the position i,j corresponds to the distance between the i-th test vector and the j-th gallery vector.
        """

        self.k = k
        self.metric_function = metric_function

        self.gallery = None
        self.labels = None
        self.num_labels = None

    def fit(self, gallery: ArrayLike, labels: ArrayLike):
        self.gallery = gallery
        self.labels = labels
        self.num_labels = self.labels.max() + 1

    def predict_proba(self, vectors: ArrayLike, groups: Optional[ArrayLike] = None) -> NDArray:
        """
        Returns the estimated probability of each vector belonging to each class.
        Because KNN works in terms of 'votes', the probability estimate of a point belonging to class i is the ratio of votes to that class.
        A vote is cast if a member of that class is among the k nearest neighbors.

        In the case multiple inputs are grouped together (such as the case where multiple images come from the same subject) the distance from the group to a point/gallery vector is used.
        The distance from the group to a point is the minimum of the distances between the group samples and the point.

        Arguments:
            vectors -- the inputs where we apply the prediction. Assumed to be a rank-2 array where each row is a point in the test set.
            groups -- (optional) this vector indicates which group each row belongs to. Groups are represented by an integer index. If not passed, each row is assumed to have its own group.
        """

        # first compute the distances
        distances = self.metric_function(self.gallery, vectors)
        if groups is not None:
            distances = self._group_distances(distances, groups)

        # get labels of the nearest k neighbors
        indices = np.argpartition(distances, self.k, -1)[:, :self.k]
        nearest_labels = np.take_along_axis(self.labels.T, indices, -1)

        # compute the ratio of votes to each label
        return self._get_probabilities(nearest_labels)
        

    def _get_probabilities(self, nearest_labels: ArrayLike) -> NDArray:
        probabilities = np.zeros([nearest_labels.shape[0], self.num_labels])
        for i in range(self.num_labels):
            probabilities[:, i] = np.sum(nearest_labels == i, axis=1)

        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def _group_distances(self, distances: NDArray, groups: NDArray) -> NDArray:
        num_groups = groups.max() + 1

        new_distances = np.empty([num_groups, distances.shape[1]])
        for i in range(num_groups):
            new_distances[i, :] = np.min(distances[groups==i, :], axis=0)

        return new_distances


def euclidean_metric(train_vectors, test_vectors):
    return np.sum((test_vectors[:, np.newaxis, :] - train_vectors[np.newaxis, :, :])**2, 2)

def cosine_metric(train_vectors, test_vectors):
    out = np.sum((test_vectors[:, np.newaxis, :]*train_vectors[np.newaxis, :, :]), 2) + EPS
    out /= np.linalg.norm(train_vectors, axis=1)[np.newaxis, :] * np.linalg.norm(test_vectors, axis=1)[:, np.newaxis] + EPS
    return 1. - out

