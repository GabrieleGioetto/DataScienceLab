from kMeans import KMeans
import pandas as pd
from silhouette import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt


def plot_silhouette_scores(samples):
    y = np.sort(samples)
    plt.plot(y)
    plt.show()


with open("2D_gauss_clusters.csv") as file:
    df = pd.read_csv(file)

    points = df.values

    clustering = KMeans(n_clusters=15)
    clustering.fit_predict(X=points)

    df["cluster"] = clustering.labels

    samples = silhouette_samples(X=points, labels=clustering.labels)
    plot_silhouette_scores(samples)

with open("chameleon_clusters.csv") as file:
    df = pd.read_csv(file)

    points = df.values

    clustering = KMeans(n_clusters=6)
    clustering.fit_predict(X=points)

    df["cluster"] = clustering.labels

    samples = silhouette_samples(X=points, labels=clustering.labels)
    plot_silhouette_scores(samples)
