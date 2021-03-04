import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

pd.set_option('display.max_columns', None)

stocks = pd.read_csv('news.csv')

stock_list = stocks['symbol'].drop_duplicates().values.tolist()
stock_dict = {}
for i in stock_list:
    ii = stocks[stocks['symbol'] == str(i)]['Title'].values.tolist()
    stock_dict[str(i)] = ii


def pre_process(text):
    text = text.lower()
    text = re.sub("</?.*?>", " <> ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    rwords = nltk.word_tokenize(text)
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(x) for x in rwords]
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    return filtered_words


vader = SentimentIntensityAnalyzer()
score_dict = {}
for key in stock_dict.keys():
    print(key)
    news = stock_dict.get(str(key))
    scores = [vader.polarity_scores(i) for i in news]
    scores = pd.DataFrame(scores)['compound'].mean()
    score_dict[str(key)] = scores
sent_select = pd.DataFrame([score_dict], index=['score']).T
sent_select = sent_select[sent_select['score'] > 0.25].index.tolist()
print(sent_select)

################# Need to improve
newthing_select = []
for key in stock_dict.keys():
    value = ' '.join(stock_dict.get(str(key)))
    filtered_words = pre_process(value)
    if 'new' in filtered_words:
        newthing_select.append(key)
# print(newthing_select)
