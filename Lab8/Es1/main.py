import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


def create_initial_dataset(f):
    tr = 20
    n_samples = 100
    X = np.linspace(-tr, tr, n_samples)
    y = f(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    y_test = y_test[X_test.argsort()]
    X_test.sort()
    return X, y, X_train, X_test, y_train, y_test


def f1(X):
    return X * np.sin(X) + 2 * X


def f2(X):
    return 10 * np.sin(X) + X ** 2


def f3(X):
    return np.sign(X) * (X ** 2 + 300) + 20 * np.sin(X)


# 1
X, y, X_train, X_test, y_train, y_test = create_initial_dataset(f2)

# 2
plt.plot(X, y)
plt.show()

# 3
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
plt.plot(X, y)
plt.plot(X_test, y_pred)
plt.show()

reg = make_pipeline(PolynomialFeatures(5), LinearRegression())
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# 4
print(f"r2 score: {r2_score(y_test, y_pred)}")

# 5
