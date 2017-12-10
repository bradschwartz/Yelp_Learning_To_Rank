#!/usr/bin/env python3
import pandas as pd
import numpy as np
import load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
import pdb

PREFIX = '../playdata/'

def textify(df):
    text_df = pd.DataFrame()
    text_df['text'] = []
    for column in df.columns:
        text_df.text = text_df.text.map(str) + ' ' + df[column].map(str)
    text_df['business_id'] = df.business_id.copy()
    return text_df


def merge(files, fill_na):
    df = load.load_data(PREFIX + files[0])
    for file in files[1:]:
        file_df = load.load_data(PREFIX + file)
        # Dropping stars since this is how we will judge our accuracy
        if file_df.get('stars') is not None:
            file_df.drop('stars', axis=1, inplace=True)
        df = df.merge(file_df, on='business_id', how='outer')

    if fill_na is not None:
        df.fillna(fill_na, inplace=True)
    stars = df.stars.copy()
    df.drop('stars', axis=1, inplace=True)

    return df, stars

def truth_sort(df, stars):
    """
    This will sort a dataframe, with the top values being the most hightly
    ranked values, by review account to break ties
    """
    df['stars'] = stars.copy()
    sort_df = df.sort_values(by=['stars', 'review_count'],
                          ascending=False)
    df.drop('stars', axis=1, inplace=True)
    return sort_df

def close_to(arr, val=0.0):
    return np.abs(arr - val).min()

def prediction_sort(text_df, predictions, pipe, classifier):
    text_df['predictions'] = predictions
    by = ['predictions']
    ascending = [False]

    if classifier == 'log':
        decision_func = pipe.decision_function(text_df.text)
        decision_func = np.array([close_to(pred, 0.0) for pred in decision_func])
        text_df['decision_func'] = decision_func
        by.append('decision_func')
        ascending.append(True)
    
    sort_df = text_df.sort_values(by=by, ascending=ascending)
    sort_df.reset_index(drop=True, inplace=True)
    return sort_df


def cross_val(df, text_df, stars, pipeline, scoring, files, print_info):
    scores = cross_val_score(pipeline, text_df.text, stars.map(str),
                             cv=None, n_jobs=-1, scoring=scoring)

    if print_info:
        unique = df.business_id.unique()
        num_unique = len(unique)
        print("Number of business:", len(df))
        print("Unique businesses: Number: {} Percentage: {}:".format(num_unique,
                                                                 round(100.0*num_unique/len(df), 3)))
        print("Used files:", ', '.join(files))
        print("Scores:{} Average:{}".format(scores, round(np.mean(scores), 3)))

def main():
    np.random.seed(10000)
    files = ['business.json', 'tip.json','review.json', 'checkin.json' ] # 'photos.json']
    df, stars = merge(files, 0)   
    text_df = textify(df)
    num_clusters = len(set(stars))

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(text_df, stars, df)

    # At this point, have df of all information about a business. Now, lets start doing ML!

    # sgd_grid_parameters = {'penalty': ('l2', 'elasticnet'), 'alpha': (0.00001, 0.000001)}
    # multi_grid_parameters = {'alpha':[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1, 0.5, 1, 2]}
    # GridSearchCV(clf, grid_parameters, verbose=1, n_jobs=1)

    # (Classifer, Scoring Method)
    classifiers = [(LogisticRegression(max_iter=100), None),
                    (SGDClassifier(max_iter=100), None),
                    # (MiniBatchKMeans(n_clusters=num_clusters, max_iter=10), 'adjusted_rand_score'),
                    (MultinomialNB(), None),
                    (DecisionTreeClassifier(), None),
                    # (MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,5)), None)
                    ]

    for clf, scoring in classifiers:
        pipe = Pipeline([('tfid', TfidfVectorizer()),
                        ('clf', clf)])
        pipe.fit(X_train.text, y_train.map(str))
        predictions = pipe.predict(X_test.text)
        # conf_mtx = confusion_matrix(y_test.map(str), predictions)
        print(clf.__module__)
        print("\tAccuracy:", 100.0 * accuracy_score(y_test.map(str), predictions))
        print("\tF1-Score:", f1_score(y_test.map(str), predictions, average='weighted'))
        print()

    # cross_val(df_train, X_train, y_train, pipe, None, files, True)
    # cross_val(df, text_df, stars, pipe, scoring, files, True)

    # pipe.fit(X_train.text, y_train.map(str))
    # predictions = pipe.predict(X_test.text)
    # print(100.0 * accuracy_score(y_test.map(str), predictions))


    # pipe.fit(text_df.text, stars.map(str))

    return df




if __name__ == '__main__':
    main()