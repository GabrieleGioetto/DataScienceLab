import pandas as pd
import matplotlib.pyplot as plt


def getIdCellByLonLat(lon, lat, minLon, minLat, number_of_grids):
    latGrid = (lat - minLat) / (maxLat - minLat) * number_of_grids
    lonGrid = (lon - minLon) / (maxLon - minLon) * number_of_grids

    idCell = lonGrid + latGrid * number_of_grids
    return idCell


with open("pois_all_info") as f_all_info:
    with open("ny_municipality_pois_id.csv") as f_ids:
        data = pd.read_csv(f_all_info, sep='\t')
        ids = pd.read_csv(f_ids, names=["ids"])

        data = pd.merge(data, ids, how="inner", left_on="@id", right_on="ids")

        data.drop("ids", 1, inplace=True)
        for column in data.columns:
            print(f"For column {column} there are {len(data[data[column].isna()])} missing values")

        percentage = 0.1
        topData = {}
        for column in ["amenity", "shop", "public_transport", "highway"]:
            dataColumn = data.groupby(column).count()["@id"].copy()
            dataColumn.sort_values(inplace=True, ascending=False)
            number_of_values = len(dataColumn)
            if number_of_values > 10:
                number_of_values = int(percentage * len(dataColumn))
            dataColumn = dataColumn[:number_of_values]
            topData[column] = dataColumn
            dataColumn.plot(kind="bar")
            plt.show()

        topAmenityData = data[data["amenity"].isin(topData["amenity"].index)].copy()
        topShopData = data[data["shop"].isin(topData["shop"].index)].copy()
        topPublicTransportData = data[data["public_transport"].isin(topData["public_transport"].index)].copy()
        topHighwayData = data[data["highway"].isin(topData["highway"].index)].copy()

        topData = pd.concat([topAmenityData, topShopData, topPublicTransportData, topHighwayData])

        plt.scatter(x=topAmenityData["@lon"], y=topAmenityData["@lat"], marker=".", zorder=1, color="blue")
        plt.scatter(x=topShopData["@lon"], y=topShopData["@lat"], marker=".", zorder=1, color="red")
        plt.scatter(x=topPublicTransportData["@lon"], y=topPublicTransportData["@lat"], marker=".", zorder=1,
                    color="black")
        plt.scatter(x=topHighwayData["@lon"], y=topHighwayData["@lat"], marker=".", zorder=1, color="pink")

        axes = plt.gca()
        x_lim_scatter = axes.get_xlim()
        y_lim_scatter = axes.get_ylim()

        minLon, maxLon = x_lim_scatter
        minLat, maxLat = y_lim_scatter

        ext = [x_lim_scatter[0], x_lim_scatter[1], y_lim_scatter[0], y_lim_scatter[1]]

        img = plt.imread("New_York_City_Map.PNG")
        implot = plt.imshow(img, zorder=0, extent=ext)
        aspect = img.shape[0] / float(img.shape[1]) * ((ext[1] - ext[0]) / (ext[3] - ext[2]))

        plt.gca().set_aspect(aspect)
        plt.show()

        number_of_grids = 15  # grid 15 X 15
        topData["gridId"] = getIdCellByLonLat(topData["@lon"], topData["@lat"], minLon, minLat, number_of_grids)
        topData["gridId"] = topData["gridId"].astype(int)

        groupedDataByCell = topData.groupby(["gridId"]).count().drop(["@lat", "@lon", "@type"], axis=1)
        print(groupedDataByCell)
