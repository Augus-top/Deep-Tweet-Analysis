from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from kutilities.layers import AttentionWithContext, Attention, MeanOverTime
import pandas as pd
import regex as re
import numpy as np
import pickle
import training as tr
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


MAX_SEQUENCE_LENGTH = 50

def findCorrectClasses(y_values):
    y_classes = []
    for classes in y_values:
        for i, class_values in enumerate(classes):
            if class_values == 1:
                y_classes.append(i)
                break
    return y_classes

def loadFile(name):
    with open(name, 'rb') as input:
        wi = pickle.load(input)
        return wi


# model = load_model('Categorical_Attention_LSTN_3_Epoch_20K_No_Scope_50_Batch_50_WORD_Length.h5')
model = load_model('Categorical_Simple_LSTN_3_Epoch_20K_No_Scope_50_Batch_50_WORD_Length.h5', custom_objects={"Attention": Attention()})

word_index = loadFile('word_index.pkl')
# print(word_index['positive_emoticon'])
# print(word_index['negative_emoticon'])
test_df = pd.read_csv('Test_Big.csv', sep=';', encoding='utf-8', error_bad_lines=False)
test_values = test_df['tweet_text'].values
labels_to_delete = []
prep_test = tr.preprocess_texts(test_values, False, False, labels_to_delete)
y_test = test_df["sentiment"].values
# y_test = np.delete(y_test, labels_to_delete)
test_sequences = tr.transformSequenceToInt(prep_test, word_index)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")


y_test = np_utils.to_categorical(y_test, 2)
#
# print(y_test)
y_classes = findCorrectClasses(y_test)
# print(y_classes)
# #
predictions = model.predict(test_data)
# print(predictions)



sess = tf.Session()
with sess.as_default():
    indexes = tf.argmax(predictions, axis=1)
    # print(indexes.eval())
    print('Matriz de Confusão')
    print(confusion_matrix(y_classes, indexes.eval()))

# #
# # rounded = [round(x[0]) for x in predictions]
# #
# # print(rounded)
# #
score = model.evaluate(test_data, y_test, verbose=1)
print(score[1])





# history = loadFile('history.pkl')
# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'train_val'], loc='upper left')
# plt.show()