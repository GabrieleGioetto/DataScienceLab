import math
import numpy as np
from collections import Counter


class KNearestNeighbors:
    def __init__(self, k, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        """
        Store the 'prior knowledge' of you model that will be used
        to predict new labels.
        :param X : input data points, ndarray, shape = (R,C).
        :param y : input labels, ndarray, shape = (R,).
        """
        self.X_train = X
        self.y_train = y
        allLabels = set(self.y_train)
        self.labelsIndex = {label: i for i, label in enumerate(allLabels)}  # Assegno indici a labels
        print(self.y_train)

    def predict(self, X):
        """Run the KNN classification on X.
        :param X: input data points, ndarray, shape = (N,C).
        :return: labels : ndarray, shape = (N,).
        """
        predictedLabels = ["" for i in range(len(X))]
        euclidean_distances = np.zeros((2, len(X), len(self.X_train)))
        for i, elementTest in enumerate(X):
            for j, (elementTrain, label) in enumerate(zip(self.X_train, self.y_train)):
                euclidean_distances[0, i, j] = self._euclideanDistance(elementTest, elementTrain)
                euclidean_distances[1, i, j] = self.labelsIndex[label]

            nearestKElementsIndexes = euclidean_distances[0, i, :].argsort()[:self.k]
            nearestKElementsLabels = euclidean_distances[1, i, nearestKElementsIndexes]
            predictedLabels[i] = self._getMostFrequentElement(nearestKElementsLabels)

        return predictedLabels

    def _getMostFrequentElement(self, lista):
        mostCommonValue = (Counter(lista).most_common()[0][0])

        return self._get_label_from_index(mostCommonValue)

    def _get_label_from_index(self, val):
        for key, value in self.labelsIndex.items():
            if val == value:
                return key

        return 0

    def _euclideanDistance(self, p, q):
        # somma = sum([(x - y) ** 2 for x, y in zip(p, q)])
        # return math.sqrt(somma)

        # numpy version
        return ((p - q) ** 2).sum() ** 0.5

    def _cosineDistance(self, p, q):
        somma = sum([x * y for x, y in zip(p, q)]) / \
                (sum([x ** 2 for x in p]) * sum([y ** 2 for y in q]))
        return 1 - abs(somma)

    def _manhattanDistance(self, p, q):
        somma = sum([abs(x - y) for x, y in zip(p, q)])
        return somma
