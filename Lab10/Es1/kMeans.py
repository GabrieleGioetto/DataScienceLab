import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def calculate_distance(self, X, N, centroids, n_clusters):
        distances = np.zeros((N, n_clusters))

        for j, centroid in enumerate(centroids):
            distances[:, j] = np.linalg.norm(X - centroid, axis=1)

        return distances

    def fit_predict(self, X):
        """Run the K-means clustering on X.
        :param X: input data points, array, shape = (N,C).
        :return: labels : array, shape = N.
        """
        N = len(X)
        self.labels = np.empty(N, dtype=np.int)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        i = 0
        while i < self.max_iter:
            # idx = np.random.randint(N, size=self.n_clusters)

            distances = self.calculate_distance(X, N, self.centroids, self.n_clusters)

            self.labels = np.argmin(distances, axis=1)

            for j in range(self.n_clusters):
                self.centroids[j] = np.mean(X[np.argwhere(self.labels == j)], axis=0)

            i += 1
