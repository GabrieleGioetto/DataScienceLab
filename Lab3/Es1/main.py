import csv
import pandas as pd
import mlxtend as mlx
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules



print(csv.__file__)

with open("online_retail.csv") as f:
    next(csv.reader(f))
    data = list(csv.reader(f))

    print(len(data))
    data = [d for d in data if d[0][0] != "C"]
    print(len(data))

    Invoices = {}
    # Creo dizionario del tipo : {"idInvoice" : ["item1", "item2"]}
    for purchase in data:
        if purchase[0] not in Invoices.keys():
            Invoices[purchase[0]] = []
        Invoices[purchase[0]].append(purchase[2])  # purchase 2: item name

    ItemsCompleteList = set([d[2] for d in data])
    numberOfItems = len(ItemsCompleteList)
    numberOfInvoices = len(Invoices.keys())
    print(numberOfInvoices)
    print(numberOfItems)
    paMatrix = [[0] * numberOfItems for i in range(numberOfInvoices)]

    ItemsCompleteList = sorted(ItemsCompleteList)
    # print(paMatrix[0])
    for i, key in enumerate(Invoices.keys()):
        for j, item in enumerate(ItemsCompleteList):
            if item in Invoices[key]:
                paMatrix[i][j] = 1

    print("Fine Ciclo")

    df = pd.DataFrame(data=paMatrix, columns=ItemsCompleteList)

    fi = fpgrowth(df, 0.01)
    print(len(fi))
    print(fi)
    print("--------------------------------")
    print(fi.to_string())
    print("--------------------------------")


    # Un itemset  (292  0.024565  (2656, 3003, 1599)) è composto da tre elementi ed ha supporto > 0.02
    ar = association_rules(fi, metric="confidence", min_threshold=0.85)
    print(ar)

    print("Fine")
