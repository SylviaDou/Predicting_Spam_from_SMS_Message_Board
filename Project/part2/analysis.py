'''
CSE163
Authors: Winnie Shao, Yihui Zhang, Wenjia Dou

This file perform machine learning on the dataset by using 4
different algorithms: DecisionTree, NaiveBayes, SVM, and
RandomForest. Also, it finds out the most important features for
predicting the dataset target.
'''


import preprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median
sns.set()


def question1_report(message):
    """
    pre: takes in the message dataframe.
    post: cleans the data, plots the text length histogram,
          and plots the wordcloud for both ham and spam messages.
    """
    preprocess.plot_text_length_histogram(message)
    message_ham = message[message['spam'] == 0].copy()
    message_spam = message[message['spam'] == 1].copy()
    preprocess.word_cloud(message_ham, 'ham_message')
    preprocess.word_cloud(message_spam, 'spam_message')


def question2_report(message):
    """
    pre: takes in the message dataframe.
    post: returns the dataset with five features.
    """
    message = preprocess.feature_word_count(message)
    message = preprocess.feature_average_word_length(message)
    message = preprocess.feature_url_count(message)
    message = preprocess.feature_special_punc_count(message)
    message_tfidf = preprocess.feature_tfidf(message)
    return message, message_tfidf


def features_and_labels(data, data_tfidf):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: returns the training sets features_train and labels_train, and test
          sets features_test and labels_test.
    """
    labels = data['spam']
    features = hstack((data_tfidf, np.array(data['word count'])[:, None],
                      np.array(data['average word length'])[:, None],
                      np.array(data['url count'])[:, None],
                      np.array(data['spc'])[:, None])).A
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    return features_train, features_test, labels_train, labels_test


def decision_tree_max_depth(features_train, features_test, labels_train,
                            labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: finds the best max depth for the Decision Tree model.
    """
    accuracies = []
    for i in range(1, 25):
        model = DecisionTreeClassifier(max_depth=i)
        model.fit(features_train, labels_train)
        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)
        accuracies.append({'max depth': i, 'test accuracy': test_acc})
    accuracies = pd.DataFrame(accuracies)
    sns.relplot(kind='line', x='max depth', y='test accuracy', data=accuracies)
    plt.title('Decision Tree Test Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel('Test Accuracy')
    plt.savefig('Decision Tree Max Depth.png', bbox_inches='tight')


def decision_tree_model(features_train, features_test, labels_train,
                        labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: prints the accuracy score of the Decision Tree model.
    """
    model = DecisionTreeClassifier(max_depth=12)
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    acc = accuracy_score(labels_test, predictions)
    print('Decision Tree:', acc)


def naive_bayes_model(features_train, features_test, labels_train,
                      labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: prints the accuracy score of Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    acc = accuracy_score(labels_test, predictions)
    print('Naive Bayes:', acc)


def svm_model(features_train, features_test, labels_train, labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: prints the accuracy score of the SVM model.
    """
    model = svm.SVC(kernel='linear')
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    acc = accuracy_score(labels_test, predictions)
    print('SVM:', acc)


def random_forest_max_depth(features_train, features_test,
                            labels_train, labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: finds the best max depth for the Random Forest model.
    """
    accuracies = []
    for i in range(1, 70):
        model = RandomForestClassifier(max_depth=i)
        model.fit(features_train, labels_train)
        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)
        accuracies.append({'max depth': i, 'test accuracy': test_acc})
    accuracies = pd.DataFrame(accuracies)
    sns.relplot(kind='line', x='max depth', y='test accuracy', data=accuracies)
    plt.title('Random Forest Test Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel('Test Accuracy')
    plt.savefig('Random Forest Max Depth.png', bbox_inches='tight')


def random_forest_model(features_train, features_test,
                        labels_train, labels_test):
    """
    pre: takes in the message dataframe as data and TFIDF matrix as data_tfidf.
    post: prints the accuracy score of the Random Forest model.
    """
    model = RandomForestClassifier(max_depth=60)
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    acc = accuracy_score(labels_test, predictions)
    print('Random Forest:', acc)


def important_features(feature, label, feature_name):
    """
    pre: takes in the feature, the target and feature's name.
    post: prints the median of the model accuracy and plots the
          accuracy distribution.
    """
    accuracy = []
    for i in range(0, 100):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(feature, label, test_size=0.2)
        model = svm.SVC(kernel='linear')
        model.fit(features_train, labels_train)
        predictions = model.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
    m = median(accuracy)
    print(feature_name + ' accuracy: ' + str(m))
    plt.hist(accuracy)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    name = feature_name + ' Accuracy Distribution'
    plt.title(name)
    name = feature_name + '.png'
    plt.axvline(m, 0, 20)
    plt.savefig(name)
    plt.close()


def main():
    message = pd.read_csv('sms.csv', encoding='latin-1')
    message = preprocess.clean_data(message)
    question1_report(message)
    message, message_tfidf = question2_report(message)
    # Question3: find the best model
    features_train, features_test, labels_train, labels_test = \
        features_and_labels(message, message_tfidf)
    decision_tree_max_depth(features_train, features_test,
                            labels_train, labels_test)
    random_forest_max_depth(features_train, features_test,
                            labels_train, labels_test)
    decision_tree_model(features_train, features_test, labels_train,
                        labels_test)
    naive_bayes_model(features_train, features_test, labels_train,
                      labels_test)
    svm_model(features_train, features_test, labels_train, labels_test)
    random_forest_model(features_train, features_test, labels_train,
                        labels_test)
    # Question4: important feature
    important_features(message['word count'].to_numpy().reshape(-1, 1),
                       message['spam'], 'word count')
    important_features(message['average word length'].to_numpy().
                       reshape(-1, 1),
                       message['spam'], 'average word length')
    important_features(message['url count'].to_numpy().reshape(-1, 1),
                       message['spam'], 'url count')
    important_features(message['spc'].to_numpy().reshape(-1, 1),
                       message['spam'], 'special punctuation count')
    important_features(message_tfidf, message['spam'],
                       'TFIDF')


if __name__ == '__main__':
    main()
