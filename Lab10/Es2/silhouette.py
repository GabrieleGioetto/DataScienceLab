import numpy as np


def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """
    N = len(X)
    a = np.zeros(N)
    b = np.zeros(N)

    for i in range(N):
        module_of_Ci = np.shape(X[np.argwhere(labels == labels[i])])[0]
        a[i] = 1 / (module_of_Ci - 1) * np.sum(np.linalg.norm(X[i, :] - X[np.argwhere(labels == labels[i]), :], axis=0))

    for i in range(N):
        module_of_Ck = np.shape(X[np.argwhere(labels != labels[i])])[0]
        current_label = labels[i]

        labels_set_without_current = np.unique(np.delete(labels, np.argwhere(labels == current_label)))

        dist = np.array([(1 / module_of_Ck) * np.sum(np.linalg.norm(X[i, :] - X[np.argwhere(labels == label), :], axis=0)) for label in
                         labels_set_without_current])
        b[i] = np.min(dist)

    a_b_union = np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1)

    s = (b - a) / (np.max(a_b_union, axis=1))

    return s

def silhouette_score(X, labels):
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    """
    return np.mean(silhouette_samples(X, labels))
