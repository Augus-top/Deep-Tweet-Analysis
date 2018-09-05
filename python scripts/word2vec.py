from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
import pandas as pd
import regex as re
import training as tr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

EMBEDDING_DIMENSION = 300

train_df = pd.read_csv('ColetaSemEscopo.csv', sep=';', encoding='utf-8', error_bad_lines=False)
test_values = train_df['tweet_text'].values
prep_texts = tr.preprocess_texts(test_values, False, False)
print(prep_texts)

# train model
model = Word2Vec(prep_texts, size=EMBEDDING_DIMENSION, min_count=1, workers=4, sg=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
# print(words)
# access vector for one word
w1 = ['lula']
print('Palavras mais próximas à ' + w1[0])
print(model.wv.most_similar(positive=w1, topn=6))
# save model
model.save('model.bin')

## load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
# w1 = ['lula']
# print('Palavras mais próximas à ' + w1[0])
# print(new_model.wv.most_similar(positive=w1, topn=6))


# def tsne_plot(model):
#     "Creates and TSNE model and plots it"
#     labels = []
#     tokens = []
#
#     for word in model.wv.vocab:
#         tokens.append(model[word])
#         labels.append(word)
#
#     tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#     new_values = tsne_model.fit_transform(tokens)
#
#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
#
#     plt.figure(figsize=(16, 16))
#     for i in range(len(x)):
#         plt.scatter(x[i], y[i])
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()
#
# tsne_plot(new_model)

# X = new_model[new_model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(new_model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()