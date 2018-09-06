from nltk.tokenize import WordPunctTokenizer
from nltk import RSLPStemmer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, MaxoutDense, Bidirectional, Embedding, GaussianNoise, Activation
from keras.models import Model, Sequential
from keras.constraints import maxnorm
from kutilities.layers import AttentionWithContext, Attention, MeanOverTime
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
import numpy as np
import pickle
import random

epochs = 10
batch = 50
chance_to_remove_emoticons = 95
WV_DIM = 300
MAX_SEQUENCE_LENGTH = 50

special_tokens = ['user', 'positive_emoticon', 'number', 'time', 'hashtag', 'negative_emoticon', 'exc', 'int', 'url']
re_url = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', re.MULTILINE|re.UNICODE)
re_user = re.compile('@[a-zA-Z0-9_]+')
re_hash = re.compile('#[a-zA-Z0-9]+')
re_time = re.compile('[0-9]+:[0-9]+')
re_acceptedChars = re.compile('[^A-Za-z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ_ ]+')
re_number = re.compile('[0-9]+')
vocab_size = Counter()
tokenizer = WordPunctTokenizer()
stemmer = RSLPStemmer()

def removeRepeatedCharacters(text, repeate_tolerance):
    current_char = ''
    repeated_counter = 1
    new_string = []
    for c in text:
        if c == current_char:
            if repeated_counter < repeate_tolerance:
                repeated_counter = repeated_counter + 1
                new_string.append(c)
        else:
            new_string.append(c)
            repeated_counter = 1
            current_char = c
    return ''.join(new_string)

def decideIfWillRemoveEmoticon(t):
    if t != 'positive_emoticon' and t!= 'negative_emoticon':
        return True
    return random.randrange(1, 100) > chance_to_remove_emoticons


def clear_emoticons(text):
    cleared_text = list(filter(decideIfWillRemoveEmoticon, text))
    # print(cleared_text)
    return cleared_text


def clear_text(text):
    # print(text)
    text = re_url.sub(' URL ', text)
    text = re_user.sub(' USER ', text)
    text = re_hash.sub(' HASH ', text)
    text = re_time.sub(' TIME ', text)
    text = re_number.sub(' NUMBER ', text)
    text = text.replace(':)', ' POSITIVE_EMOTICON ')
    text = text.replace(':D', ' POSITIVE_EMOTICON ')
    text = text.replace(':P', ' POSITIVE_EMOTICON ')
    text = text.replace(':(', ' NEGATIVE_EMOTICON ')
    text = removeRepeatedCharacters(text, 2)
    text = text.replace('!', ' EXC ')
    text = text.replace('?', ' INT ')
    text = re_acceptedChars.sub('', text)
    text = tokenizer.tokenize(text)
    text = [t.lower() for t in text]
    return text


def is_useful_text(text):
    found_emoticon = ''
    special_token_counter = 0
    for word in text:
        for token in special_tokens:
            if (word == token):
                special_token_counter = special_token_counter + 1;
        if found_emoticon == '' and (word == 'positive_emoticon' or word == 'negative_emoticon'):
            found_emoticon = word
        if (found_emoticon == 'positive_emoticon' and word == 'negative_emoticon') or (found_emoticon == 'negative_emoticon' and word == 'positive_emoticon'):
            return False
        if word == 'rt':
            return False
    if special_token_counter == len(text):
        return False
    return True

def preprocess_texts(text_values, stem_flag, useful_text_flag, labels_to_delete):
    prepared_texts = []
    counter = 0
    for i, text in enumerate(tqdm(text_values)):
        prep_text = clear_text(text)
        if useful_text_flag and not  is_useful_text(prep_text):
            counter = counter + 1
            labels_to_delete.append(i)
            continue
        prep_text = clear_emoticons(prep_text)
        if stem_flag:
            prep_text = [stemmer.stem(t) for t in prep_text]
        vocab_size.update(prep_text)
        prepared_texts.append(prep_text)
    # print('Removidos ' + str(counter))
    # print('Restou ' + str(len(prepared_texts)))
    return prepared_texts


def transformSequenceToInt(texts, word_index):
    sequence = []
    for text in tqdm(texts):
        seq_int = []
        for word in text:
            seq_int.append(word_index.get(word, 0))
        sequence.append(seq_int)
    return sequence


def buildEmbeddingLayer(word_vectors, word_index, dimension):
    nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
    # we initialize the matrix with random numbers
    wv_matrix = (np.random.rand(nb_words, dimension) - 0.5) / 5.0
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            # words not found in embedding index will be all-zeros.
            wv_matrix[i] = embedding_vector
        except:
            pass
    wv_layer = Embedding(nb_words,
                         dimension,
                         mask_zero=False,
                         weights=[wv_matrix],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False)
    return wv_layer


def constructSimpleModel(model_name, categorical, wv_layer, train_data, y, number_classes):

    if categorical:
        loss = 'categorical_crossentropy'
        y = np_utils.to_categorical(y, number_classes)
        output_layer_size = number_classes
    else:
        loss = 'binary_crossentropy'
        output_layer_size = 1

    # print('Shape of data tensor', train_data.shape)
    print('Shape of label tensor', y.shape)
    print(loss)
    print(output_layer_size)

    # Inputs
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = wv_layer(comment_input)

    # biLSTM
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)

    # Output
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    preds = Dense(output_layer_size, activation='sigmoid')(x)

    # build the model
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss=loss,
                  optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                  metrics=['accuracy'])

    # Train and SaveModel
    hist = model.fit(train_data, y, validation_split=0.1, epochs=epochs, batch_size=batch, shuffle=True, verbose=2)
    model.save(model_name)
    return hist


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0., l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, dropout=dropout_U, kernel_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def prepareAttentionModel(embeddings, classes, max_length, unit=LSTM, cells=64, layers=1, **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)
    model = Sequential()
    model.add(embeddings)

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        model.add(get_RNN(unit, cells, bi, return_sequences=rs,
                          dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))

    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def constructAttentionModel(model_name, wv_layer, train_data, y, number_classes):

    loss = 'categorical_crossentropy'
    y = np_utils.to_categorical(y, number_classes)
    output_layer_size = number_classes

    # print('Shape of data tensor', train_data.shape)
    print('Shape of label tensor', y.shape)
    print(loss)
    print(output_layer_size)

    model = prepareAttentionModel(wv_layer, classes=number_classes, max_length=MAX_SEQUENCE_LENGTH,
                                   unit=LSTM, layers=2, cells=150,
                                   bidirectional=True,
                                   attention="simple",
                                   noise=0.3,
                                   final_layer=False,
                                   dropout_final=0.5,
                                   dropout_attention=0.5,
                                   dropout_words=0.3,
                                   dropout_rnn=0.3,
                                   dropout_rnn_U=0.3,
                                   clipnorm=1, lr=0.001, loss_l2=0.0001, )
    print('construiu')
    hist = model.fit(train_data, y, validation_split=0.1, epochs=epochs, batch_size=batch, shuffle=True, verbose=2)
    model.save(model_name)
    return hist


if __name__ == "__main__":

    train_df = pd.read_csv('Train.csv', sep=';', encoding='utf-8', error_bad_lines=False)
    # tweet_values = list(train_df['tweet_text'].fillna("NAN_WORD").values)

    tweet_values = train_df['tweet_text'].values
    labels_to_delete = []
    prep_texts = preprocess_texts(tweet_values, False, True, labels_to_delete)
    print("The vocabulary contains {} unique tokens".format(len(vocab_size)))
    new_model = Word2Vec.load('model.bin')
    print(new_model)
    word_vectors = new_model.wv
    MAX_NB_WORDS = len(word_vectors.vocab)
    word_index = {t[0]: i+1 for i, t in enumerate(vocab_size.most_common(MAX_NB_WORDS))}
    # print(word_index['positive_emoticon'])

    train_sequences = transformSequenceToInt(prep_texts, word_index)
    data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
    # print(data[0])

    y = train_df["sentiment"].values
    y = np.delete(y, labels_to_delete)

    wv_layer = buildEmbeddingLayer(word_vectors, word_index, WV_DIM)

    # Save Word_Index
    with open('word_index.pkl', 'wb') as output:
        pickle.dump(word_index, output, pickle.HIGHEST_PROTOCOL)

    # constructSimpleModel('Categorical_Simple_LSTN_3_Epoch_20K_No_Scope.h5', True, wv_layer, data, y, 2)
    # constructSimpleModel('Binary_Simple_LSTN_3_Epoch_20K_No_Scope.h5', False, wv_layer, data, y, 2)
    history = constructSimpleModel('Categorical_Simple_LSTN_3_Epoch_20K_No_Scope_50_Batch_50_WORD_Length.h5', True, wv_layer, data, y, 2)
    # history = constructAttentionModel('Categorical_Attention_LSTN_3_Epoch_20K_No_Scope_50_Batch_50_WORD_Length.h5', wv_layer, data, y, 2)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'train_val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'train_val'], loc='upper left')
    plt.show()
    #Save history
    with open('history.pkl', 'wb') as output:
        pickle.dump(history.history, output, pickle.HIGHEST_PROTOCOL)