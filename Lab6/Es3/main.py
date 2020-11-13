import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
from myRandomForestClassifier import MyRandomForestClassifier

# 1
train = pd.read_csv("mnist_train.csv", header=None)
test = pd.read_csv("mnist_test.csv", header=None)

X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]  # labels

X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]  # labels

# 2
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)
print(f" Accuracy score of prediction (DecisionTree): {accuracy_score(y_test, y_predictions)}")

# 3/4
p = len(X_train.columns)
my_clf = MyRandomForestClassifier(n_estimators=20, max_features=int(math.sqrt(p)))
my_clf.fit(X_train, y_train)
y_predictions = my_clf.predict(X_test)
print(f" Accuracy score of prediction (MyRandomTree) : {accuracy_score(y_test, y_predictions)}")

# 5
clf = RandomForestClassifier(n_estimators=20, max_features=int(math.sqrt(p)))
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)
print(f" Accuracy score of prediction (Sklearn RandomTree): {accuracy_score(y_test, y_predictions)}")

# 6
print(my_clf.compute_feature_importance())
