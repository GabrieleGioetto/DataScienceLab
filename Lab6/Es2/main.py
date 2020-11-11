import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

with open("2d-synthetic.csv") as f:
    data = pd.read_csv(f)

    data.set_index("label", inplace=True)
    plt.scatter(data.loc[0, "x0"], data.loc[0, "x1"], c="red")
    plt.scatter(data.loc[1, "x0"], data.loc[1, "x1"], c="blue")

    plt.show()

    data.reset_index(inplace=True)

    classifier = DecisionTreeClassifier()
    classifier.fit(data[["x0", "x1"]], data["label"])

    dot_code = export_graphviz(classifier, feature_names=["x0", "x1"])
    print(dot_code)
