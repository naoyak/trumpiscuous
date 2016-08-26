'''
Script to build corpora
'''

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import History, ModelCheckpoint
from collections import Counter
import numpy as np
import random
import sys
from nltk import word_tokenize

def prep_data(filenames, cutoff_ratio=0.25, maxlen=15, step=3):
    text = ''
    for f in filenames:
        text += open('corpora/{}'.format(f)).read().lower()

    print('corpus length:', len(text))

    # break into sentences
    words_split = word_tokenize(text)
    word_counter = Counter(words_split).most_common()

    top_words = [w[0] for w in word_counter[:int(len(word_counter) * cutoff_ratio)]]
    words_split = [word for word in words_split if word in top_words]

    word_set = sorted(list(set(words_split)))
    print('total words:', len(word_set))
    word_indices = dict((w, i) for i, w in enumerate(word_set))
    indices_word = dict((i, w) for i, w in enumerate(word_set))

    word_seqs = []
    next_words = []

    for i in range(0, len(words_split) - maxlen, step):
        word_seqs.append(words_split[i: i + maxlen])
        next_words.append(words_split[i + maxlen])

    return (maxlen, step, words_split, word_seqs, next_words, word_set, word_indices, indices_word)



def build_data(filenames, cutoff_ratio=0.25, maxlen=15, step=3):
    _, _, words_split, word_seqs, next_words, word_set, word_indices, indices_word = prep_data(filenames, cutoff_ratio, maxlen, step)

    print('Vectorization...')
    X = np.zeros((len(word_seqs), maxlen, len(word_set)), dtype=np.bool)
    print(X.shape)
    y = np.zeros((len(word_seqs), len(word_set)), dtype=np.bool)
    for i, word_seq in enumerate(word_seqs):
        for t, word in enumerate(word_seq):
            X[i, t, word_indices[word]] = 1
        y[i, word_indices[next_words[i]]] = 1
    return (X, y, maxlen, step, words_split, word_seqs, next_words, word_set, word_indices, indices_word)


def new_model(maxlen, word_set):
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(word_set))))
    model.add(Dropout(0.2))
    model.add(Dense(len(word_set)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train_model(corpus_data, save_path, model_path=None, batch_size=128, nb_epoch=1):
    X, y, maxlen, step, words_split, word_seqs, next_words, word_set, word_indices, indices_word = corpus_data
    if model_path is None:
        model = new_model(maxlen, word_set)
    else:
        model = load_model(model_path)

    checkpoint = ModelCheckpoint(filepath=save_path,
        verbose=1, save_best_only=False)
    history = History()

    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[checkpoint, history])
    return model

def generate_speech(model, diversity, corpus_data, length=200):
    maxlen, step, words_split, word_seqs, word_set, word_indices, indices_word = corpus_data
    start_index = random.randint(0, len(word_seqs))

    print('----- diversity:', diversity)

    generated = []
    sentence = word_seqs[start_index]
    generated.extend(sentence)
    print('----- Generating with seed: "' + " ".join(sentence) + '"')
    # sys.stdout.write(" ".join(generated))

    for i in range(length):
        x = np.zeros((1, maxlen, len(word_set)))
        for t, word in enumerate(sentence):
            x[0, t, word_indices[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        generated.append(next_word)
        sentence = sentence[1:]
        sentence.append(next_word)
        # sys.stdout.write((" " if next_word[0].isalnum() else "") + next_word)
        # sys.stdout.flush()
        # print()

    generated_sentence = "".join([" " + word if word[0].isalnum() else word for word in generated])
    return generated_sentence
