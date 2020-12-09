import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import gmaps
import gmaps.datasets
from ipywidgets.embed import embed_minimal_html
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def create_index_column(df_dev, name_column):
    df_grouped = df_dev.groupby(name_column).mean()["price"]
    df_grouped = df_grouped.sort_values()
    d = {"price": df_grouped, "new_index": np.arange(1, len(df_grouped) + 1)}
    df_grouped_with_new_index = pd.DataFrame(data=d)
    return df_grouped_with_new_index.loc[df_dev[name_column], "new_index"].to_numpy()


def show_price_heatmap_in_gmaps(df_dev):
    # df_dev = df_dev[df_dev["neighbourhood"] == "Upper East Side"]
    locations = df_dev[['latitude', 'longitude']]
    weights = df_dev['price']
    fig = gmaps.figure()
    heatmap_layer = gmaps.heatmap_layer(locations, weights=weights)
    fig.add_layer(heatmap_layer)
    embed_minimal_html('export.html', views=[fig])


def plot_grouped_by_column(df_dev, column_name):
    df_grouped_by = df_dev.groupby(column_name).mean()["price"]
    df_grouped_by.plot(y="price", use_index=True, kind="bar")
    plt.show()


with open("development.csv") as dev_file:
    df_dev = pd.read_csv(dev_file)

    # show_price_heatmap_in_gmaps(df_dev)

    print(len(df_dev["neighbourhood"].unique()))
    print(df_dev["neighbourhood"].unique())

    print(df_dev.iloc[0])
    # print(df_dev[df_dev["price"] > 5000]["neighbourhood"])

    for column_name in ["neighbourhood_group", "neighbourhood", "room_type", "minimum_nights", "number_of_reviews", "last_review",
                        "reviews_per_month", "calculated_host_listings_count", "availability_365"]:
        plot_grouped_by_column(df_dev, column_name)

    # Ordinare neighbourhood hood per costo medio e categorizzarli

    df_dev["neighbourhood_index"] = create_index_column(df_dev, "neighbourhood")
    # df_dev["neighbourhood_group_index"] = create_index_column(df_dev, "neighbourhood_group")
    # df_dev["room_type_index"] = create_index_column(df_dev, "room_type")
    df_dev = pd.get_dummies(df_dev, columns=["room_type", "neighbourhood_group"])

    print(df_dev)
    print(df_dev.columns)

    useful_columns = df_dev.columns.values[14:]

    # TODO: creare mappa 20*20 delle coordinate e creare categorical da lì
    # TODO: creare categorie group quartier e tipo appartamento
    # TODO: fare i test direttamente sui quartieri e non sui gruppi di quartieri
    

    X_train, X_test, y_train, y_test = train_test_split(df_dev[useful_columns], df_dev["price"], test_size=0.2)

    print(X_train.columns)

    reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    print(r2)

    # show_price_heatmap_in_gmaps(df_dev)
