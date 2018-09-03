from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import pandas as pd
import regex as re
import numpy as np
import pickle
import training as tr

MAX_SEQUENCE_LENGTH = 350

def loadWordIndex():
    with open('word_index.pkl', 'rb') as input:
        wi = pickle.load(input)
        return wi


model = load_model('rnn_model.h5')

word_index = loadWordIndex()
# print(word_index['positive_emoticon'])
# print(word_index['negative_emoticon'])
test_df = pd.read_csv('Test.csv', sep=',', encoding='utf-8', error_bad_lines=False)
test_values = test_df['tweet_text'].values
prep_test = tr.preprocess_texts(test_values)
test_sequences = tr.transformSequenceToInt(prep_test, word_index)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
y_test = test_df["sentiment"].values

# Y_test = np_utils.to_categorical(y_test, 2)
# print(Y_test)
#
# predictions = model.predict(test_data)
# print(predictions)
#
# rounded = [round(x[0]) for x in predictions]
#
# print(rounded)
#
# score = model.evaluate(test_data, y_test, verbose=1)
# print(score)