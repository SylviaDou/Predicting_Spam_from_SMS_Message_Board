'''
CSE163
Authors: Winnie Shao, Yihui Zhang, Wenjia Dou

A file implements various data analysis functions to clean
the dataset, do the exploratory analysis and process data
to be machine-readable by doing natural language processing.
'''


import matplotlib.pyplot as plt
import wordcloud
import numpy as np
import re
import string
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def clean_data(message):
    """
    pre: takes in message dataframe.
    post: cleans the dataframe.
    """
    message = message.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    message = message.rename(columns={"v1": "label", "v2": "text"})
    message['spam'] = message['label'].map({'spam': 1, 'ham': 0}).astype(int)
    message['total char'] = message['text'].apply(len)
    return message


def plot_text_length_histogram(message):
    """
    pre: takes in the message dataframe.
    post: plots the length distribution histogram for both spam and ham
          message.
    """
    spam = message[message['spam'] == 1]
    ham = message[message['spam'] == 0]
    bins = np.linspace(-40, 500, 100)
    plt.hist(spam['total char'], bins, alpha=0.5, label='spam')
    plt.hist(ham['total char'], bins, alpha=0.5, label='ham')
    plt.legend(loc='upper right')
    plt.xlabel('Total Char')
    plt.ylabel('Frequency')
    plt.title('The Histogram of Total Char of Ham and Spam Message')
    plt.savefig('test_length_histogram.png')
    plt.close()


def word_cloud(message, title):
    """
    pre: takes in the sms dataframe and a title for the plot.
    post: plots the wordcloud for certain all the text in the
          data frame.
    """
    text = ' '.join(message['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,
                                        background_color='lightgrey',
                                        colormap='plasma',
                                        width=800,
                                        height=600).generate(text)
    plt.figure(figsize=(10, 7), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=20)
    name = title + '.png'
    plt.savefig(name)
    plt.close()


def feature_word_count(message):
    """
    pre: takes in the message dataset.
    post: returns the new message dataset with word count feature.
    """
    message['word count'] = 0
    for i in range(len(message)):
        text = message.loc[i, 'text']
        count = 0
        terms = text.split()
        for term in terms:
            if (not term.lower().startswith('http')) \
                    and re.sub(r'\W+', '', term) != '':
                count = count + 1
        message.loc[i, 'word count'] = count
    return message


def feature_average_word_length(message):
    """
    pre: takes in the message dataset.
    post: returns the new message dataset with average word length feature.
    """
    message['average word length'] = 0
    for i in range(len(message)):
        text = message.loc[i, 'text']
        count = 0
        word_count = 0
        terms = text.split()
        for term in terms:
            if term.lower()[0:4] != 'http' and re.sub(r'\W+', '', term) != '':
                count = count + len(re.sub(r'\W+', '', term))
                word_count = word_count + 1
        if word_count != 0:
            message.loc[i, 'average word length'] = count / word_count
        else:
            message.loc[i, 'average word length'] = 0
    return message


def feature_url_count(message):
    """
    pre: takes in message dataframe.
    post: returns the dataframe with feature: url count.
    """
    message['url count'] = message['text'].str.split(). \
        apply(lambda l: [x for x in l if x.lower().startswith('http')])
    message['url count'] = message['url count'].apply(len)
    return message


def feature_special_punc_count(message):
    '''
    pre: takes in message dataframe.
    post: returns the dataframe with feature: spc(special punctuation count).
    '''
    message['spc'] = message['text'].astype(str). \
        apply(lambda x: re.sub(r'[a-zA-z]+', '', x))
    message['spc'] = message['spc']. \
        apply(lambda x: re.sub(r'(?<=\D)[.,]|[.,](?=\D)', '', x))
    message['spc'] = message['spc'].apply(lambda x: re.sub(r'\d+', '', x))
    message['spc'] = message['spc'].str.split(). \
        apply(lambda x: len(''.join(x)))
    return message


def remove_punc_stop(message):
    """
    pre: takes in the message dataframe.
    post: removes the punctuation and stopwords in the text.
    """
    sms_np = [char for char in message if char not in string.punctuation]
    sms_np = "".join(sms_np).split()
    sms_npns = []
    for word in sms_np:
        if word.lower() not in stopwords.words("english"):
            sms_npns.append(word.lower())
    return sms_npns


def feature_tfidf(message):
    """
    pre: takes in the message dataframe.
    post: returns the TFIDF matrix for all the text in the dataframe.
    """
    bow_transformer = CountVectorizer(analyzer=remove_punc_stop)\
        .fit(message['text'])
    bow_data = bow_transformer.transform(message['text'])
    tfidf_transformer = TfidfTransformer().fit(bow_data)
    message_tfidf = tfidf_transformer.transform(bow_data)
    return message_tfidf
