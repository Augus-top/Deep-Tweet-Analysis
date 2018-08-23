from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from tqdm import tqdm
from gensim.models import Word2Vec
import pandas as pd
import regex as re

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

train_df = pd.read_csv('Coleta.csv', sep=',', encoding='utf-8', error_bad_lines=False)
test_values = train_df['tweet_text'].values
prep_texts = preprocess_texts(test_values)
print("The vocabulary contains {} unique tokens".format(len(vocab_size)))
# print(prep_texts)

# train model
model = Word2Vec(prep_texts, size=100, min_count=1, workers=4, sg=1)
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
# model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)