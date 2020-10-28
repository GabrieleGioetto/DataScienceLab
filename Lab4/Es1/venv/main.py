import numpy as np
import pandas as pd
from kNearestNeighbors import KNearestNeighbors

df = pd.read_csv(
    "iris.csv",
    header=None,
)

print(f"# of total tuples in df: {len(df)}")

dfSampled = df.sample(frac=0.2)

X_test = dfSampled.iloc[:, 0:4].to_numpy().copy()
y_test = dfSampled.iloc[:, 4].to_numpy().copy()

X_train = df.loc[df.index.difference(dfSampled.index)].iloc[:, 0: 4].to_numpy().copy()  # Remaining 80% values
y_train = df.loc[df.index.difference(dfSampled.index)].iloc[:, 4].to_numpy().copy()  # Remaining 80% values

# print(X_train)
# print(y_train)

knn = KNearestNeighbors(10)
knn.fit(X_train, y_train)
y_pred = (knn.predict(X_test))

print(y_test)
print(y_pred)

countTrue = 0
for i in range(len(y_test)):
    countTrue += y_test[i] == y_pred[i]
    print(f"{y_test[i] == y_pred[i]}  {y_test[i]} {y_pred[i]}")

print(countTrue)
