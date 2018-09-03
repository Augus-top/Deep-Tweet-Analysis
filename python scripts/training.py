from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import pandas as pd
import regex as re
import numpy as np
import pickle
import random

epochs = 3
chance_to_remove_emoticons = 95

re_url = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', re.MULTILINE|re.UNICODE)
re_user = re.compile('@[a-zA-Z0-9_]+')
re_hash = re.compile('#[a-zA-Z0-9]+')
re_time = re.compile('[0-9]+:[0-9]+')
re_punc = re.compile('[.,;]++')
re_number = re.compile('[0-9]+')
vocab_size = Counter()
tokenizer = WordPunctTokenizer()

def decideIfWillRemoveEmoticon(t):
    if t != 'positive_emoticon' and t!= 'negative_emoticon':
        return True
    return random.randrange(1, 100) > chance_to_remove_emoticons


def clear_emoticons(text):
    cleared_text = list(filter(decideIfWillRemoveEmoticon, text))
    # print(cleared_text)
    return cleared_text


def clean_text(text):
    # print(text)
    text = re_url.sub('URL', text)
    text = re_user.sub('USER', text)
    text = re_hash.sub('HASH', text)
    text = re_time.sub('TIME', text)
    text = re_number.sub('NUMBER', text)
    text = re_punc.sub('', text)
    text = text.replace(':)', 'POSITIVE_EMOTICON')
    text = text.replace(':D', 'POSITIVE_EMOTICON')
    text = text.replace(':P', 'POSITIVE_EMOTICON')
    text = text.replace(':(', 'NEGATIVE_EMOTICON')
    text = tokenizer.tokenize(text)
    text = [t.lower() for t in text]
    text = clear_emoticons(text)
    vocab_size.update(text)
    return text


def preprocess_texts(text_values):
    prepared_texts = []
    for text in tqdm(text_values):
        prep_text = clean_text(text)
        prepared_texts.append(prep_text)
    return prepared_texts


def transformSequenceToInt(texts, word_index):
    sequence = []
    for text in tqdm(texts):
        seq_int = []
        for word in text:
            seq_int.append(word_index[word])
        sequence.append(seq_int)
    return sequence


if __name__ == "__main__":

    train_df = pd.read_csv('Train.csv', sep=',', encoding='utf-8', error_bad_lines=False)
    # tweet_values = list(train_df['tweet_text'].fillna("NAN_WORD").values)

    tweet_values = train_df['tweet_text'].values
    prep_texts = preprocess_texts(tweet_values)

    # total_tweets = preprocess_texts(tweet_values + test_values)

    print("The vocabulary contains {} unique tokens".format(len(vocab_size)))
    new_model = Word2Vec.load('model.bin')
    print(new_model)
    word_vectors = new_model.wv
    MAX_NB_WORDS = len(word_vectors.vocab)
    MAX_SEQUENCE_LENGTH = 350
    word_index = {t[0]: i+1 for i, t in enumerate(vocab_size.most_common(MAX_NB_WORDS))}
    # print(word_index['positive_emoticon'])

    train_sequences = transformSequenceToInt(prep_texts, word_index)
    # print(test_sequences)
    # get(t, 0) for t in total_tweets] for tweet in total_tweets[:len(tweet_values)]]
    # test_sequences = [[word_index.get(t, 0) for t in total_tweets] for tweet in total_tweets[len(test_values):]]
    data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

    y = train_df["sentiment"].values
    print('Shape of data tensor', data.shape)
    print('Shape of label tensor', y.shape)

    WV_DIM = 300
    nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
    # we initialize the matrix with random numbers
    wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            # words not found in embedding index will be all-zeros.
            wv_matrix[i] = embedding_vector
        except:
            pass

    # Save Word_Index
    with open('word_index.pkl', 'wb') as output:
        pickle.dump(word_index, output, pickle.HIGHEST_PROTOCOL)


    wv_layer = Embedding(nb_words,
                         WV_DIM,
                         mask_zero=False,
                         weights=[wv_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False)

    # Inputs
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = wv_layer(comment_input)

    # biLSTM
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)

    # Output
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    preds = Dense(1, activation='sigmoid')(x)

    # build the model
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                  metrics=['accuracy'])

    # Train and SaveModel
    model.fit([data], [y], validation_split=0.1, epochs=epochs, batch_size=256, shuffle=True, verbose=1)
    model.save('rnn_model.h5')
