import string
import math
import csv


def tokenize(docs):
    """Compute the tokens for each document.
    Input: a list of strings. Each item is a document to tokenize.
    Output: a list of lists. Each item is a list containing the tokens of the
    relative document.
    """
    tokens = []
    for doc in docs:
        for punct in string.punctuation:
            doc = doc.replace(punct, " ")
        split_doc = [token.lower() for token in doc.split(" ") if token]
        tokens.append(split_doc)
    return tokens


with open("aclimdb_reviews_train.csv") as f:
    data = list(csv.reader(f))
    data = data[1:]
    # print(data[:10])

    reviews = [d[0] for d in data]
    reviewsTokenized = tokenize(reviews)

    listWordsCount = []  # DF for every document

    for i, review in enumerate(reviewsTokenized):

        listWordsCount.append({})

        for word in review:
            if word not in listWordsCount[i]:
                listWordsCount[i][word] = 1
            else:
                listWordsCount[i][word] += 1

    print(listWordsCount[0])
    N = len(reviews)

    DFt = {}
    for wordsCount in listWordsCount:
        for word in wordsCount.keys():
            if word not in DFt:
                DFt[word] = 1
            else:
                DFt[word] += 1

    IDFt = {}
    for word in DFt.keys():
        IDFt[word] = math.log(N / DFt[word])

    # print(sorted(list(IDFt.values()))[-10:])

    TF_IDF = []
    for i, wordsCount in enumerate(listWordsCount):
        TF_IDF.append({})
        for word in wordsCount.keys():
            TF_IDF[i][word] = wordsCount[word] * IDFt[word]
