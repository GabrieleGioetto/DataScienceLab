from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

# 1
X, y = make_regression(n_samples=2000, n_features=100, random_state=50)

# 2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=50, shuffle=True)
reg = LinearRegression()

reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)

print(f"r2-score: {r2_score(y_test, y_test_pred)}")

# 3
X, y = make_regression(n_samples=2000, random_state=42, noise=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=50, shuffle=True)
reg = LinearRegression()

reg.fit(X_train, y_train)
y_test_pred = reg.predict(X_test)

print(f"r2-score: {r2_score(y_test, y_test_pred)}")

print(f"Coefficienti della regression lineare: \n{reg.coef_}")
