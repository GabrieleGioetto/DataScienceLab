import csv


def getTopHottestMeasurements(data, City, N):
    dataCity = list(filter(lambda d: d[3] == City, data))

    topTemperatures = list((map(lambda d: d[1], dataCity)))
    topTemperatures.sort(reverse=True)
    return topTemperatures[:N]


def getTopColdestMeasurements(data, City, N):
    dataCity = list(filter(lambda d: d[3] == City, data))

    topTemperatures = list((map(lambda d: d[1], dataCity)))
    topTemperatures.sort()
    return topTemperatures[:N]


with open("GLT_filtered.csv") as f:
    data = list(csv.reader(f))
    print(data[0])
    print(data[1])
    data = data[1:]

    precedentValue = 0
    nextValue = 0

    for i in range(0, len(data) - 1):
        if data[i][1] == "":
            j = i + 1
            while (j < len(data) - 1) and (data[j][1] == ""):
                j += 1
            if j == len(data) - 1:
                nextValue = 0
            else:
                nextValue = float(data[j][1])

            data[i][1] = str((precedentValue + nextValue) / 2)

        precedentValue = float(data[i][1])

    print(getTopHottestMeasurements(data, "Abidjan", 10))
