from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
import pandas as pd
import regex as re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

re_url = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', re.MULTILINE|re.UNICODE)
re_user = re.compile('@[a-zA-Z0-9_]+')
re_hash = re.compile('#[a-zA-Z0-9]+')
re_time = re.compile('[0-9]+:[0-9]+')
re_punc = re.compile('[.,;]++')
re_number = re.compile('[0-9]+')
vocab_size = Counter()
tokenizer = WordPunctTokenizer()

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
    vocab_size.update(text)
    # print(text)
    return text

def preprocess_texts(text_values):
    prepared_texts = []
    for text in tqdm(text_values):
        prep_text = clean_text(text)
        prepared_texts.append(prep_text)
    return prepared_texts

# train_df = pd.read_csv('ColetaSemEscopo.csv', sep=',', encoding='utf-8', error_bad_lines=False)
# test_values = train_df['tweet_text'].values
# prep_texts = preprocess_texts(test_values)
# print("The vocabulary contains {} unique tokens".format(len(vocab_size)))
# # print(prep_texts)
#
# # train model
# model = Word2Vec(prep_texts, size=300, min_count=1, workers=4, sg=1)
# # summarize the loaded model
# print(model)
# # summarize vocabulary
# words = list(model.wv.vocab)
# # print(words)
# # access vector for one word
# w1 = ['presidente']
# print('Palavras mais próximas à ' + w1[0])
# print(model.wv.most_similar(positive=w1, topn=6))
# # save model
# model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
# w1 = ['presidente']
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