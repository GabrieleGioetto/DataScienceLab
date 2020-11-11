from sklearn.datasets import load_wine
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

dataset = load_wine()
X = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]

# 1
print(f"# of record in X {len(X)}")
print(f"# of record in y {len(y)}")

print(np.argwhere(np.isnan(X)))  # No null values

# 2
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# 3
dot_code = export_graphviz(classifier, feature_names=feature_names)
# print(dot_code)

# 4
y_predictions = classifier.predict(X)
print(f" Accuracy score of prediction: {accuracy_score(y, y_predictions)}")  # 1.0

# 5
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# 6
classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)
print(f" Accuracy score of prediction: {accuracy_score(y_test, y_predictions)}")  # circa 0.9

# 7
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 2, 4, 8],
    "splitter": ["best", "random"]
}

configs = ParameterGrid(params)

for config in configs:
    clf = DecisionTreeClassifier(**config)
    clf.fit(X_train, y_train)
    y_predictions = classifier.predict(X_test)
    print(config)
    print(f" Accuracy score of prediction: {accuracy_score(y_test, y_predictions)}")  # circa 0.9

# 8
