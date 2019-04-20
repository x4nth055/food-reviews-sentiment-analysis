import numpy as np
import pandas as pd
import tqdm
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

from utils import clean_text, tokenize_words
from config import N, test_size

def load_review_data():
    df = pd.read_csv("data/Reviews.csv")
    # preview
    print(df.head())
    print(df.tail())
    vocab = []
    X = np.zeros((len(df), 2), dtype=object)
    for i in tqdm.tqdm(range(len(df)), "Cleaning X"):
        target = df['Text'].loc[i]
        X[i, 0] = clean_text(target)
        X[i, 1] = df['Score'].loc[i]
        for word in X[i, 0].split():
            vocab.append(word)

    # vocab = set(vocab)
    vocab = Counter(vocab)

    # delete words that occur less than 10 times
    vocab = { k:v for k, v in vocab.items() if v >= N }

    # word to integer encoder dict
    vocab2int = {word: i for i, word in enumerate(vocab, start=1)}

    # pickle int2vocab for testing 
    print("Pickling vocab2int...")
    pickle.dump(vocab2int, open("data/vocab2int.pickle", "wb"))

    # encoded reviews
    for i in tqdm.tqdm(range(X.shape[0]), "Tokenizing words"):
        X[i, 0] = tokenize_words(X[i, 0], vocab2int)

    lengths = [ len(row)  for row in X[:, 0] ]
    print("min_length:", min(lengths))
    print("max_length:", max(lengths))

    X_train, X_test, y_train, y_test = train_test_split(X[:, 0], X[:, 1], test_size=test_size, shuffle=True, random_state=19)

    return X_train, X_test, y_train, y_test, vocab

