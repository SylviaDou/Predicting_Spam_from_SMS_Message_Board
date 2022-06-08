'''
CSE163
Authors: Winnie Shao, Yihui Zhang, Wenjia Dou

A file that contains tests for preprocess.py.
'''


import preprocess
import pandas as pd
from cse163_utils import assert_equals

df = pd.read_csv('sample_data.csv')


def test_feature_word_count():
    """
    post: tests the feature_word_count method in question1.py.
    """
    assert_equals(4, preprocess.feature_word_count(df).loc[0, 'word count'])
    assert_equals(2, preprocess.feature_word_count(df).loc[1, 'word count'])
    assert_equals(1, preprocess.feature_word_count(df).loc[2, 'word count'])
    assert_equals(3, preprocess.feature_word_count(df).loc[3, 'word count'])
    assert_equals(2, preprocess.feature_word_count(df).loc[4, 'word count'])


def test_feature_average_word_length():
    """
    post: tests the feature_average_word_length method in question1.py.
    """
    assert_equals(14 / 4, preprocess.feature_average_word_length(df)
                  .loc[0, 'average word length'])
    assert_equals(5 / 2, preprocess.feature_average_word_length(df)
                  .loc[1, 'average word length'])
    assert_equals(4, preprocess.feature_average_word_length(df)
                  .loc[2, 'average word length'])
    assert_equals(19 / 3, preprocess.feature_average_word_length(df)
                  .loc[3, 'average word length'])
    assert_equals(7 / 2, preprocess.feature_average_word_length(df)
                  .loc[4, 'average word length'])


def test_feature_url_count():
    """
    post: check feature_url_count function returned against expected,
          throws an AssertionError if they don't match.
    """
    assert_equals(0, preprocess.feature_url_count(df).loc[0, 'url count'])
    assert_equals(1, preprocess.feature_url_count(df).loc[4, 'url count'])


def test_feature_special_punc_count():
    '''
    post: check feature_special_punc_count function returned against expected,
          throws an AssertionError if they don't match.
    '''
    assert_equals(6, preprocess.feature_special_punc_count(df).loc[2, 'spc'])
    assert_equals(2, preprocess.feature_special_punc_count(df).loc[0, 'spc'])
    assert_equals(4, preprocess.feature_special_punc_count(df).loc[4, 'spc'])


def main():
    test_feature_word_count()
    test_feature_average_word_length()
    test_feature_url_count()
    test_feature_special_punc_count()


if __name__ == '__main__':
    main()
