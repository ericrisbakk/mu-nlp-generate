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


# Data generator to avoid memory issues.
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    looped = False
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, EMBEDDING_SIZE), dtype=np.float32)
        y = np.zeros((batch_size, len(vocab)), dtype=np.bool)
        for i in range(batch_size):
            # For each word in the sentence fragment, get the vector
            for t, w in enumerate(sentence_list[index]):
                x[i, t, :] = wv[w]
            # Set the appropriate y-value.
            y[i, word_indices[next_word_list[index]]] = 1
            # Each batch does a different sentence.
            index = index + 1
            # Reset the index at the end.
            if index == len(sentence_list):
                index = 0
                looped = True
        # Stopping condition: If we have gone around, stop yielding.
        if looped:
            return None
        else:
            yield x, y


# From https://github.com/enriqueav/lstm_lyrics/blob/master/lstm_train_embedding.py
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1. - (percentage_test / 100.)))
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


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences + sentences_test))
    seed = (sentences + sentences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" " + next_word)
        examples_file.write('\n')
    examples_file.write('=' * 80 + '\n')
    examples_file.flush()


"""
*******************************************************
                          GLOBAL VARS
*******************************************************
"""

# Saving results
RESULT_FOLDER = "Results/"
MODEL_NAME = "FinalModelSimple5.h5"

# Gensim
EMBEDDING_SIZE = 100
WINDOW_SIZE = 7

# Dataset prep
SEQUENCE_LEN = 5
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
data = pd.read_csv("../LocalData/ProcessedSongData.csv")
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
print("Loading Keyed Vectors.")
wv = KeyedVectors.load("../LocalData/song_word_vec.kv")

# Confirm that all words in vocab is in KeyedVectors
for word in vocab_keys:
    if not word in wv:
        print("Word not found: ", word)

print("Keyed Vectors Loaded.")

print("Pairing word and index.")
# Create word-index pairing.
# 0 is reserved.
word_indices = dict((c, i + 1) for i, c in enumerate(vocab_keys))
indices_word = dict((i + 1, c) for i, c in enumerate(vocab_keys))

# Writing a dictionary to file.
print("Writing the word-token dict to a file.")
f = open("../LocalData/tokenizer_dict.txt", "w")
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
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])

print('Sequences:', len(sentences))

# Split into training and test set.
# x, y, x_test, y_test
(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

# Model!
print("Creating model.")
model = Sequential()
# We will convert any text to a word embedding before sending it on its way.
model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, EMBEDDING_SIZE)))
model.add(Dropout(DROPOUT))
# Classification of next word.
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print("Model created.")

file_path = "../checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
            "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
            (len(vocab_keys), SEQUENCE_LEN, 0)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_acc', patience=20)
callbacks_list = [checkpoint, print_callback, early_stopping]

examples_file = open("../LocalData/examples.txt", "w")

print("Fitting model.")
model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                    steps_per_epoch=int(len(sentences) / BATCH_SIZE) + 1,
                    epochs=100,
                    callbacks=callbacks_list,
                    validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                    validation_steps=int(len(sentences_test) / BATCH_SIZE) + 1)

model.save("../LocalData/" + MODEL_NAME)