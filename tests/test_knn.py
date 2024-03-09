import pytest
import numpy as np

from knn import KNNClassifier

NUM_POINTS = 20


@pytest.fixture
def gallery():
    gallery = np.arange(NUM_POINTS)
    gallery = np.stack([gallery, np.zeros(NUM_POINTS)], 1)
    return gallery

@pytest.fixture
def labels():
    labels = np.arange(NUM_POINTS)[:, np.newaxis]
    return labels

def dummy_metric(a, b):
    return np.abs(a[np.newaxis, :, 0] - b[:, 0, np.newaxis])


def test_knn(gallery, labels):
    """
    Test the KNNClassifier class using dummy data and a dummy metric function.
    Test in the not grouped case.
    """

    points = np.array([[0, 0], [1, 0]])

    expected = np.zeros((2, NUM_POINTS))
    expected[:2, 0] = 1/3
    expected[:2, 1] = 1/3
    expected[:2, 2] = 1/3

    knn = KNNClassifier(3, dummy_metric)
    knn.fit(gallery, labels)
    output = knn.predict_proba(points)

    np.testing.assert_almost_equal(output, expected)


def test_knn_grouped(gallery, labels):
    """
    Test the KNNClassifier class using dummy data and a dummy metric function.
    Test the grouped case.
    """
    points = np.array([[0, 0], [1, 0], [10, 0], [15, 0]])
    groups = np.array([0, 1, 0, 0])

    expected = np.zeros((2, NUM_POINTS))
    expected[1, 0] = 1/3
    expected[1, 1] = 1/3
    expected[1, 2] = 1/3

    expected[0, 0] = 1/3
    expected[0, 10] = 1/3
    expected[0, 15] = 1/3

    knn = KNNClassifier(3, dummy_metric)
    knn.fit(gallery, labels)
    output = knn.predict_proba(points, groups)

    np.testing.assert_almost_equal(output, expected)

