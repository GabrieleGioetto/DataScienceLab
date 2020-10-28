import math
from _collections import defaultdict


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

    def predict(self, X):
        """Run the KNN classification on X.
        :param X: input data points, ndarray, shape = (N,C).
        :return: labels : ndarray, shape = (N,).
        """
        predictedLabels = []
        for flowerTest in X:
            euclidean_distances = []
            for flowerTrain, label in zip(self.X_train, self.y_train):
                euclidean_distances.append((self._euclideanDistance(flowerTest, flowerTrain), label))

            euclidean_distances.sort(key=lambda tup: tup[0])
            euclidean_distances = euclidean_distances[0:self.k]
            predictedLabels.append(self._getMostFrequentElement(euclidean_distances))

        return predictedLabels

    def _getMostFrequentElement(self, lista):
        freq = defaultdict(lambda: 0)
        for (flowerTrain, label) in lista:
            freq[label] += 1
        mostFrequent = self._keywithmaxval(freq)
        return mostFrequent

    def _keywithmaxval(self, d):
        """ a) create a list of the dict's keys and values;
            b) return the key with the max value"""
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    def _euclideanDistance(self, p, q):
        somma = sum([(x - y) ** 2 for x, y in zip(p, q)])
        return math.sqrt(somma)

    def _cosineDistance(self, p, q):
        somma = sum([x * y for x, y in zip(p, q)]) / (sum([x ** 2 for x in p]) * sum([y ** 2 for y in q]))
        return 1 - math.abs(somma)

    def _manhattanDistance(self, p, q):
        somma = sum([math.abs(x - y) for x, y in zip(p, q)])
        return somma
