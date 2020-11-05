import csv
import math


def get_character_to_print(characterInteger):
    if characterInteger < 64:
        return " "
    if characterInteger < 128:
        return "."
    if characterInteger < 192:
        return "*"

    return "#"


with open("mnist_test.csv") as f:
    k = 129  # Example
    db = list(csv.reader(f))
    row = db[k]
    print(row)

    rowLength = 28

    for i in range(0, rowLength):
        for j in range(0, rowLength):
            characterToPrint = get_character_to_print(int(row[rowLength * i + j]))
            print(characterToPrint, end="")

        print("")

    print("-------------------------------------------------------------")

    row1 = list(map(float, db[24]))
    row2 = list(map(float, db[29]))

    print(row1)
    euclideanDistance = math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(row1, row2)))
    print(euclideanDistance)
