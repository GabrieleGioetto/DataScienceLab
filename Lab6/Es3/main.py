import pandas as pd

train = pd.read_csv("mnist_test.csv", header=None)
test = pd.read_csv("mnist_train.csv", header=None)

X_train = train[1:]
y_train = train[0]  # labels

X_test = test[1:]
y_test = test[0]  # labels

