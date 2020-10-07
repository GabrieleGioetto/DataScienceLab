import csv
import math

# CALCOLO MEDIE
with open("iris.csv") as f:
    flowers = {}
    counterFlowers = {}
    for row in csv.reader(f):
        specie = row[4]
        if str(specie) not in flowers.keys():
            flowers[str(specie)] = [0, 0, 0, 0]
            counterFlowers[str(specie)] = 0

        flowers[str(specie)] = [sum(x) for x in zip(flowers[str(specie)], map(lambda x: float(x), row[0:4]))]
        counterFlowers[str(specie)] += 1

    flowersAverages = {}
    for specie in flowers.keys():
        flowersAverages[specie] = [x / counterFlowers[str(specie)] for x in flowers[str(specie)]]

    print(f"Averages for species {flowersAverages}")

# CALCOLO DEVIAZIONI STANDARD
with open("iris.csv") as f:
    flowers = {}
    for row in csv.reader(f):
        specie = row[4]
        if str(specie) not in flowers.keys():
            flowers[str(specie)] = [0, 0, 0, 0]

        localQuadraticSum = [(x - avg) ** 2 for x, avg in zip(map(float, row[0:4]), flowersAverages[specie])]
        flowers[str(specie)] = [sum(x) for x in zip(flowers[str(specie)], localQuadraticSum)]

    flowersDeviations = {}
    for specie in flowers.keys():
        flowersDeviations[specie] = [math.sqrt(x / counterFlowers[str(specie)]) for x in flowers[str(specie)]]

    print(f"Deviations for species {flowersDeviations}")
