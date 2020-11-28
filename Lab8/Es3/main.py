import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

with open("SummaryofWeather.csv") as f:
    df = pd.read_csv(f)

    # 2
    grouped_df = df.groupby("STA")
    sta_null_values = []
    for key, group_df in grouped_df:
        sta_null_values.append([key, group_df.isna().sum().sum()])

    df_null_values = pd.DataFrame(data=sta_null_values, columns=["STA", "null_values"])
    df_null_values_top_10 = df_null_values.sort_values(by="null_values", ascending=False).iloc[:10]

    with open("WeatherStationLocations.csv") as f2:
        df_locations = pd.read_csv(f2)

        print(df_locations[df_locations["WBAN"].isin(df_null_values_top_10["STA"])])

    # 3
    df = df[df["STA"] == 22508]

    df.loc[:, "Date"] = pd.to_datetime(df["Date"])

    # 4
    plt.plot(df["Date"], df["MeanTemp"])
    plt.show()

    # 5
    rolling_windows = []
    W = 30
    T = len(df["Date"])
    for i in range(T - W):
        rolling_windows.append([df.iloc[i]["Date"], *df.iloc[i:i + W + 1]["MeanTemp"]])

    columns = ["Date", *np.arange(W + 1)]
    rolling_windows = pd.DataFrame(rolling_windows, columns=columns)

    # 6
    X_train = rolling_windows[rolling_windows["Date"].dt.year.isin([1940, 1941, 1942, 1943])][rolling_windows.columns[1:31]]
    y_train = rolling_windows[rolling_windows["Date"].dt.year.isin([1940, 1941, 1942, 1943])][rolling_windows.columns[31]]
    X_test = rolling_windows[rolling_windows["Date"].dt.year == 1944][rolling_windows.columns[1:31]]
    y_test = rolling_windows[rolling_windows["Date"].dt.year == 1944][rolling_windows.columns[31]]

    # 7
    reg1 = LinearRegression()
    reg1.fit(X_train, y_train)
    y_test_pred = reg1.predict(X_test)

    print(f"r-2 score linear regression: {r2_score(y_test, y_test_pred)}")

    # reg2 = make_pipeline(PolynomialFeatures(4), Lasso(alpha=0.5, tol=0.2))
    # reg2.fit(X_train, y_train)
    # y_test_pred = reg2.predict(X_test)
    #
    # print(f"r-2 score linear regression: {r2_score(y_test, y_test_pred)}")

    # 8
    plt.plot(rolling_windows[rolling_windows["Date"].dt.year == 1944]["Date"], y_test)
    plt.plot(rolling_windows[rolling_windows["Date"].dt.year == 1944]["Date"], y_test_pred)
    plt.show()
