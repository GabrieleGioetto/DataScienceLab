from kMeans import KMeans
import pandas as pd
from silhouette import Silhouette

with open("2D_gauss_clusters.csv") as file:
    df = pd.read_csv(file)

    points = df.values

    clustering = KMeans(n_clusters=15)
    clustering.fit_predict(X=points)

    df["cluster"] = clustering.labels

    print(Silhouette.silhouette_score(X=points, labels=clustering.labels))
