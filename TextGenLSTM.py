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
from keras.layers import Dense, Flatten, Activation, Bidirectional, LSTM, Dropout
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
    for i, w in enumerate(s_list):
        if w == '\r\n':
            s_list[i] = '\\r\\n'
    return s_list


# Data generator to avoid memory issues.
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, EMBEDDING_SIZE), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            # For each word in the sentence fragment, get the vector
            for t, w in enumerate(sentence_list[index]):
                x[i, t, :] = wv[w]
            # Set the appropriate y-value.
            y[i] = word_indices[next_word_list[index]]
            # Each batch does a different sentence.
            index = index + 1
            # Reset the index at the end.
            if index == len(sentence_list):
                index = 0
        yield x, y


# From https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py
# Added a fraction argument in order to not select all of the data,
def shuffle_and_split_training_set(sentences_original, next_original, fraction=1, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    sample_num = int(fraction * len(sentences_original))

    print("Fraction used: ", sample_num, "/", len(sentences_original))

    tmp_sentences = []
    tmp_next_word = []

    for i in np.random.permutation(sample_num):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(sample_num * (1. - (percentage_test / 100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_interval_end(m, s_list, n_training, epoch_num, data_intervals):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Run: %d\n' % n_training)
    examples_file.write('\n----- Number of epochs per run: %d\n' % epoch_num)
    examples_file.write('\n----- Data Intervals: %d\n' % data_intervals)


    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(s_list))
    # Some sentence.
    seed = s_list[seed_index]

    print("Seed: ", seed)

    example_songs = []
    for i, diversity in enumerate([0.3, 0.4, 0.5, 0.6, 0.7]):
        # Print information and chosen seed.
        sentence = seed
        examples_file.write('\n----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        print('Example song with diversity', diversity, ':')
        example_songs.append(' '.join(sentence))
        for j in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN, EMBEDDING_SIZE), dtype=np.float32)
            for t, w in enumerate(sentence):
                x_pred[0, t, :] = wv[w]

            preds = m.predict(x_pred, verbose=0)[0]

            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            example_songs[i] += (' ' + next_word)
            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" " + next_word)

        examples_file.write('\n')
        print('Song: ', example_songs[i])
    examples_file.write('=' * 80 + '\n')
    examples_file.flush()


def generate_nn_data(sentence_list, next_word_list):
    x = np.zeros((len(sentence_list), SEQUENCE_LEN, EMBEDDING_SIZE), dtype=np.float32)
    y = np.zeros(len(next_word_list), dtype=np.int32)
    # Go through each sentence fragment
    for i, s in enumerate(sentence_list):
        # For each word in the sentence fragment, get the vector
        for t, w in enumerate(s):
            x[i, t, :] = wv[w]
        # Set the appropriate y-value.
        y[i] = word_indices[next_word_list[i]]
    return x, y


def get_model():
    new_model = Sequential()
    # We will convert any text to a word embedding before sending it on its way.
    new_model.add(LSTM(12, return_sequences=False, input_shape=(SEQUENCE_LEN, EMBEDDING_SIZE)))
    new_model.add(Dropout(DROPOUT))
    # Classification of next word.
    new_model.add(Dense(len(vocab_keys), activation='softmax'))
    return new_model


"""
*******************************************************
                          GLOBAL VARS
*******************************************************
"""

# Saving results
RESULT_FOLDER = "Results/"
MODEL_NAME = "Simple5.h5"
EXAMPLE_FILE = "./LocalData/examples.txt"

# Gensim
EMBEDDING_SIZE = 100
WINDOW_SIZE = 7

# Dataset prep
SEQUENCE_LEN = 10
STEP = 1

# NN Model
DROPOUT = 0.5
BATCH_SIZE = 100

"""
*******************************************************
                          MAIN SCRIPT
*******************************************************
"""

# Load data.
print("Loading data")
data = pd.read_csv("./LocalData/ProcessedSongData.csv")
# Ensure that "token" and "corrected" columns are lists, and not strings of list.
# When saving to csv the lists are converted into string.
print("data loaded.")

print("Starting to tokenize.")
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

print('Example song: ', clean_songs[0])

# Load the keyed vectors.
print("Loading Keyed Vectors.")
wv = KeyedVectors.load("./LocalData/song_word_vec.kv")

wv['\\r\\n'] = wv['\r\n']

# Confirm that all words in vocab is in KeyedVectors
for word in vocab_keys:
    if not word in wv:
        print("Word not found: ", word)

print("Keyed Vectors Loaded.")

print("Pairing word and index.")
# Create word-index pairing.
# 0 is reserved.
word_indices = dict((c, i) for i, c in enumerate(vocab_keys))
indices_word = dict((i, c) for i, c in enumerate(vocab_keys))

# Writing a dictionary to file.
print("Writing the word-token dict to a file.")
f = open("./LocalData/tokenizer_dict.txt", "w")
for word in vocab_keys:
    f.write(repr(word) + " " + str(word_indices[word]) + "\n")
f.close()

print("Generating sequences")
# Generate SEQUENCE_LEN words from all songs.
sentences = []
next_words = []

for song in clean_songs:
    if len(song) > SEQUENCE_LEN:
        for i in range(0, len(song) - SEQUENCE_LEN, STEP):
            sentences.append(song[i: i + SEQUENCE_LEN])
            next_words.append(song[i + SEQUENCE_LEN])

print('Sequences:', len(sentences))

# Model!
print("Creating model.")
model = get_model()

optimizer = RMSprop(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("Model created.")

examples_file = open(EXAMPLE_FILE, "w")

print("Reducing dataset for faster runtimes.")

# Split into training and test set.
# x, y, x_test, y_test
(sentences_train, next_words_train), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences,
                                                                                                        next_words,
                                                                                                        fraction=0.3)
print("Initial Sentences: ", sentences[0:1000])
print("Training sentences: ", sentences_train[0:1000])
print("Fitting model.")

RUNS_TOTAL = 10
DATA_INTERVALS = 10
EPOCHS = 1

# Get list of interval values.
interval_step = int(len(sentences_train)/DATA_INTERVALS)
INTERVALS = [i*interval_step for i in range(DATA_INTERVALS)]
INTERVALS.append(len(sentences_train))

print('Intervals:')
print(INTERVALS)

for i in range(RUNS_TOTAL):
    for j in range(DATA_INTERVALS):
        print('\n\nRUN ', str(i + 1))
        print('Interval:', INTERVALS[j], ' - ', INTERVALS[j+1], '\n\n')
        # Prepare interval data.
        sentence_interval = sentences_train[INTERVALS[j]:INTERVALS[j+1]]

        next_words_interval = next_words_train[INTERVALS[j]:INTERVALS[j+1]]
        X, Y = generate_nn_data(sentence_interval, next_words_interval)

        model.fit(x=X, y=Y, batch_size=100, epochs=EPOCHS)

    print("Generating example")
    on_interval_end(model, sentences_test, i, EPOCHS, DATA_INTERVALS)

    model.save("./LocalData/" + 'Run' + str(i) + MODEL_NAME)

print("Done fitting.")

print("Saving model")
model.save("./LocalData/" + 'Final' + MODEL_NAME)
