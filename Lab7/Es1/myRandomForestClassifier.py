from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class MyRandomForestClassifier():
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.classifiers = []
        self.total_features = 0

    # train the trees of this random forest using subsets of X (and y)
    def fit(self, X, y):
        N = len(X)
        self.total_features = len(X.columns)

        for i in range(self.n_estimators):
            print(f"Training tree n°{i}...")
            clf = DecisionTreeClassifier(max_features=self.max_features)

            X_train_sampled = X.sample(n=N, replace=True)
            y_train_sampled = y.iloc[X_train_sampled.index]

            clf.fit(X_train_sampled, y_train_sampled)
            self.classifiers.append(clf)

    # predict the label for each point in X
    def predict(self, X):
        y_predictions = np.empty((len(self.classifiers), len(X)))
        for j, clf in enumerate(self.classifiers):
            y_predictions[j] = (clf.predict(X))

        df = pd.DataFrame(y_predictions)
        mostFrequentLabels = df[:].mode()
        return mostFrequentLabels.iloc[0, :]

    def compute_feature_importance(self):
        feature_importance_for_tree = np.empty(shape=(self.n_estimators, self.total_features))
        for i, clf in enumerate(self.classifiers):
            feature_importance_for_tree[i] = clf.feature_importances_

        total_importance = feature_importance_for_tree.sum()
        return feature_importance_for_tree.sum(axis=0) / total_importance
