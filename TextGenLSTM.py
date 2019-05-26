"""
This implementation was done following a number of texts / tutorials:

- song lyrics generation: https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb,
- a continuation of the previous tutorial using word embeddings: https://medium.com/@enriqueav/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1,
- and this tutorial for word embeddings: https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/.
"""

"""
*******************************************************
                          Import Statements
*******************************************************
"""
import pandas as pd
import numpy as np
from collections import Counter

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Bidirectional, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.optimizers import RMSprop

"""
*******************************************************
                          METHODS
*******************************************************
"""

# Turn sentence into list of words.
def tokenize(s):
    s_list = [w for w in s.split(' ') if w.strip() != '' or w == '\r\n']
    return s_list



"""
*******************************************************
                          GLOBAL VARS
*******************************************************
"""

# Saving results
RESULT_FOLDER = "Results/"

# Gensim
EMBEDDING_SIZE = 100
WINDOW_SIZE = 7


"""
*******************************************************
                          MAIN SCRIPT
*******************************************************
"""


# Load data.
data = pd.read_csv("LocalData/ProcessedSongData.csv")
# Ensure that "token" and "corrected" columns are lists, and not strings of list.
# When saving to csv the lists are converted into string.
print("data loaded.")

print("Starting to tokenize.")
data["t_clean"] = data.clean.apply(tokenize)
print("Tokenized clean.")
data["t_corrected"] = data.corrected.apply(tokenize)
print("Tokenized corrected.")

# Create vocab
print("Creating vocab.")
text_values = data.t_corrected.values
vocab = Counter()

text_in_words = []
for song in text_values:
    vocab.update(song)
    text_in_words.extend(song)

print("Number of words total: ", len(text_in_words))
print("Unique words: ", len(vocab))

vocab_keys = sorted(list(vocab.keys()))

clean_songs = text_values

# Load the keyed vectors.
wv = KeyedVectors.load("LocalData/song_word_vec.kv")

# Confirm that all words in vocab is in KeyedVectors
for word in vocab_keys:
    if not word in wv:
        print("Word not found: ", word)

