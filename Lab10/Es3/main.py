import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
import string

file_names = os.listdir("T-newsgroups")
file_names = ["T-newsgroups/" + file for file in file_names]


# files = []
# for file in file_names:
#     f = open(f"T-newsgroups/{file}", "r")
#     files.append(f.read())
#     f.close()


class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            lemmas.append(lemma)
        return lemmas


lemmaTokenizer = LemmaTokenizer()

stopwords = sw.words('english') + list(string.punctuation) + ["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need',
                                                              'sha', 'wa', 'wo', 'would']
print(list(string.punctuation))
vectorizer = TfidfVectorizer(input="filename", max_df=0.9, min_df=5, tokenizer=lemmaTokenizer, stop_words=stopwords)
tfidf_X = vectorizer.fit_transform(file_names)
feature_names = vectorizer.get_feature_names()


