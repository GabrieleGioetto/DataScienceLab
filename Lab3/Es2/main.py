import itertools as it
import json


# from mlxtend.frequent_patterns import fpgrowth
# from mlxtend.frequent_patterns import association_rules


def listContainTuple(tuple, list):
    return all(element in list for element in tuple)


def firstElementsEqual(tuple1, tuple2):
    length = len(tuple1)
    for i in range(length - 1):
        if tuple1[i] != tuple2[i]:
            return False
    return True


# -------------------------------------------------------------- #
# 1° parte

data = [
    ['a', 'b'],
    ['b', 'c', 'd'],
    ['a', 'c', 'd', 'e'],
    ['a', 'd', 'e'],
    ['a', 'b', 'c'],
    ['a', 'b', 'c', 'd'],
    ['b', 'c'],
    ['a', 'b', 'c'],
    ['a', 'b', 'd'],
    ['b', 'c', 'e']
]

C1 = set([(item,) for itemList in data for item in itemList])
C1 = sorted(C1)

C = [C1]  # lista di liste di tuple
L = []  # Lista di dizionari con valori: ((tupla elementi), frequenza)
i = 0
minsup = 2

while len(C[i]) > 0:

    # CALCOLO Li
    L.append({})
    for letterTuple in C[i]:
        L[i][letterTuple] = 0
        for itemList in data:
            if listContainTuple(letterTuple, itemList):
                L[i][letterTuple] += 1
        if (L[i][letterTuple]) < minsup:  # PRUNING min support
            del L[i][letterTuple]

    # CALCOLO Ci+1
    tupleCount = len(L[i])
    for j in range(tupleCount):
        for k in range(j + 1, tupleCount):
            C.append([])
            listTuples = list(L[i].keys())
            if firstElementsEqual(listTuples[j], listTuples[k]):
                C[i + 1].append(listTuples[j] + (listTuples[k][-1],))  # Aggiungo l'ultima lettera

    for j in range(len(C[i + 1]) - 1, -1, -1):  # Ciclo al contrario per evitare errori di index nel del
        if not all(
                letterTuple in list(L[i].keys()) for letterTuple in
                list(it.combinations(C[i + 1][j], len(C[i + 1][j]) - 1))):
            del C[i + 1][j]  # PRUNING

    i += 1

print(L)
print(f"Number of elements in L: {sum([len(k) for k in L])}")
print("---------------------------------------------------------------------------------------------")

# --------------------------------------------------------------------------------------------- #
# 2° parte

with open("modified_coco.json") as f:
    data = json.load(f)

    dataAnnotations = [d["annotations"] for d in data]
    C1 = set([(item,) for itemList in dataAnnotations for item in itemList])
    C1 = sorted(C1)

    C = [C1]  # lista di liste di tuple
    L = []
    i = 0
    minsup = 0.02
    T = len(dataAnnotations)

    while len(C[i]) > 0:

        # CALCOLO Li
        L.append({})
        for letterTuple in C[i]:
            L[i][letterTuple] = 0
            for itemList in dataAnnotations:
                if listContainTuple(letterTuple, itemList):
                    L[i][letterTuple] += 1
            L[i][letterTuple] /= T  # Calculate support
            if L[i][letterTuple] < minsup:  # PRUNING min support
                del L[i][letterTuple]

        # CALCOLO Ci+1
        tupleCount = len(L[i])
        for j in range(tupleCount):
            for k in range(j + 1, tupleCount):
                C.append([])
                listTuples = list(L[i].keys())
                if firstElementsEqual(listTuples[j], listTuples[k]):
                    C[i + 1].append(listTuples[j] + (listTuples[k][-1],))  # Aggiungo l'ultima lettera

        for j in range(len(C[i + 1]) - 1, -1, -1):  # Ciclo al contrario per evitare errori di index nel del
            if not all(
                    letterTuple in list(L[i].keys()) for letterTuple in
                    list(it.combinations(C[i + 1][j], len(C[i + 1][j]) - 1))):
                del C[i + 1][j]  # PRUNING

        i += 1

    print(L)
    print("---------------------------------------------------------------------------------------------")
