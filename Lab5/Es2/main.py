import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("831394006_T_ONTIME.csv") as f:
    data = pd.read_csv(f, parse_dates=["FL_DATE"])

    # print(data)
    # print("----------------------------------------------------")
    # print(data.info())
    # print("----------------------------------------------------")
    # print(data.describe())
    # print("----------------------------------------------------")
    # print(data["FL_DATE"].min())

    # 3
    data = data[data["CANCELLED"] == 0]  # Filtro voli non cancellati

    # 4
    print(data.groupby("UNIQUE_CARRIER").count())
    print(data.groupby("UNIQUE_CARRIER").mean()["ARR_DELAY"])

    # 5
    data["weekday"] = data["FL_DATE"].dt.dayofweek
    data["delaydelta"] = data["ARR_DELAY"] - data["DEP_DELAY"]

    # 6
    print(data.groupby("weekday").mean()["ARR_DELAY"])
    data.groupby("weekday").mean()["ARR_DELAY"].plot(kind="bar")
    plt.show()

    # 7
    avg_arr_delay_by_carrier_on_weekend = data[data["weekday"] >= 5].groupby(["UNIQUE_CARRIER", "weekday"]).mean()[
        "ARR_DELAY"]
    avg_arr_delay_by_carrier_on_weekday = data[data["weekday"] < 5].groupby(["UNIQUE_CARRIER", "weekday"]).mean()[
        "ARR_DELAY"]
    print(avg_arr_delay_by_carrier_on_weekend)
    print(avg_arr_delay_by_carrier_on_weekday)

    # 8
    # indexes = [data["UNIQUE_CARRIER"], data["ORIGIN"], data["DEST"], data["FL_DATE"]]
    # dataIndexed = pd.DataFrame(data, index=[indexes])
    dataGrouped = data.set_index(["UNIQUE_CARRIER", "ORIGIN", "DEST", "FL_DATE"])
    print(data.index)
    print("--------------------------------------------")

    # 9
    print(dataGrouped.loc[["AA", "DL"], "LAX", :, :][["DEP_TIME", "DEP_DELAY"]])
    print("--------------------------------------------")

    # 10
    dataFirstWeek = dataGrouped[pd.Int64Index(dataGrouped.index.get_level_values("FL_DATE").isocalendar().week) == 1]
    print(f'MEDIA:  {dataFirstWeek.loc[:, :, "LAX", :]["ARR_DELAY"].mean()}')
    print("--------------------------------------------")

    # 11
    pivot_table = (data.pivot_table("FL_DATE", index="weekday", columns="UNIQUE_CARRIER", aggfunc="count"))
    print(pivot_table)
    pairwise_corr = pivot_table.corr()
    sns.heatmap(pairwise_corr)

    plt.show()

    # 12
    pivot_table = data.pivot_table("ARR_DELAY", index="weekday", columns="UNIQUE_CARRIER", aggfunc="mean")
    pairwise_corr = pivot_table.corr()
    sns.heatmap(pairwise_corr)

    plt.show()
