import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kMeans import KMeans
import seaborn as sns


def show_cluster(df, n_clusters):
    df.plot(kind="scatter", x="x", y="y")
    plt.show()

    points = df.values

    clustering = KMeans(n_clusters=n_clusters)
    clustering.fit_predict(X=points)

    df["cluster"] = clustering.labels

    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette="deep")
    plt.show()


with open("2D_gauss_clusters.csv") as file:
    df = pd.read_csv(file)

    show_cluster(df, n_clusters=15)

with open("chameleon_clusters.csv") as file:
    df = pd.read_csv(file)

    show_cluster(df, n_clusters=6)
