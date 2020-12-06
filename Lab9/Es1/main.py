import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def create_index_column(df_dev, name_column):
    df_grouped = df_dev.groupby(name_column).mean()["price"]
    df_grouped = df_grouped.sort_values()
    d = {"price": df_grouped, "new_index": np.arange(1, len(df_grouped) + 1)}
    df_grouped_with_new_index = pd.DataFrame(data=d)
    return df_grouped_with_new_index.loc[df_dev[name_column], "new_index"].to_numpy()


with open("development.csv") as dev_file:
    df_dev = pd.read_csv(dev_file)

    # print(df_dev.columns)

    print(df_dev.iloc[0])

    # print(df_dev["neighbourhood_group"].unique())
    # print(df_dev["neighbourhood"].unique())
    # print(df_dev["number_of_reviews"].unique())
    # print(df_dev["minimum_nights"].unique())

    # df_grouped_by_reviews = df_dev.groupby("number_of_reviews").mean()["price"]
    # df_grouped_by_ne = df_dev.groupby("neighbourhood").mean()["price"]
    # df_grouped_by_ne_group = df_dev.groupby("neighbourhood_group").mean()["price"]
    # df_grouped_by_min_nights = df_dev.groupby("minimum_nights").mean()["price"]
    # df_grouped_by_room_type = df_dev.groupby("room_type").mean()["price"]
    # df_grouped_by_rev_per = df_dev.groupby("reviews_per_month").mean()["price"]
    # df_grouped_by_ava = df_dev.groupby("availability_365").mean()["price"]
    # df_grouped_by_list_count = df_dev.groupby("calculated_host_listings_count").mean()["price"]

    # df_grouped_by_reviews.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_ne.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_ne_group.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_min_nights.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_rev_per.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_ava.plot(y="price", use_index=True, kind="bar")
    # plt.show()
    # df_grouped_by_list_count.plot(y="price", use_index=True, kind="bar")
    # plt.show()

    useful_columns = ["neighbourhood_group_index", "room_type_index", "neighbourhood_index"]

    # Ordinare neighbourhood hood per costo medio e categorizzarli

    # df_grouped_by_ne = df_grouped_by_ne.sort_values()
    # d = {"price": df_grouped_by_ne, "new_index": np.arange(1, len(df_grouped_by_ne) + 1)}
    # df_grouped_by_ne_with_new_index = pd.DataFrame(data=d)
    # df_dev["neighbourhood_index"] = df_grouped_by_ne_with_new_index.loc[df_dev["neighbourhood"], "new_index"].to_numpy()
    #
    # df_grouped_by_ne_group = df_grouped_by_ne_group.sort_values()
    # d = {"price": df_grouped_by_ne_group, "new_index": np.arange(1, len(df_grouped_by_ne_group) + 1)}
    # df_grouped_by_ne_group_with_new_index = pd.DataFrame(data=d)
    # df_dev["neighbourhood_group_index"] = df_grouped_by_ne_group_with_new_index.loc[df_dev["neighbourhood_group"], "new_index"].to_numpy()
    #
    # df_grouped_by_room_type = df_grouped_by_room_type.sort_values()
    # d = {"price": df_grouped_by_room_type, "new_index": np.arange(1, len(df_grouped_by_room_type) + 1)}
    # df_grouped_by_room_type_with_new_index = pd.DataFrame(data=d)
    # df_dev["room_type_index"] = df_grouped_by_room_type_with_new_index.loc[df_dev["room_type"], "new_index"].to_numpy()

    df_dev["neighbourhood_index"] = create_index_column(df_dev, "neighbourhood")
    df_dev["neighbourhood_group_index"] = create_index_column(df_dev, "neighbourhood_group")
    df_dev["room_type_index"] = create_index_column(df_dev, "room_type")

    X_train, X_test, y_train, y_test = train_test_split(df_dev[useful_columns], df_dev["price"], test_size=0.2)
