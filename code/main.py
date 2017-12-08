#!/usr/bin/env python3
import pandas as pd
import numpy as np
import load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
import pdb


def main():
    prefix = '../playdata/'

    # files = 'business.json'
    # business_df = load.load_data(prefix + files)
    # stars = business_df.stars
    # business_df = business_df.drop('stars', axis=1)
    # # base
    # vect = TfidfVectorizer()
    
    # df = pd.DataFrame()
    # df['text'] = []
    # for column in business_df.columns:
    #     df['text'] = df['text'].map(str) + ' ' + business_df[column].map(str)
    # vectorized = VarianceThreshold(0.0001).fit_transform(vect.fit_transform(df.text))
    
    # parameters = {'penalty': ('l2', 'elasticnet'), 'alpha': (0.00001, 0.000001)}
    # sgd = SGDClassifier(max_iter=10)
    # clf = GridSearchCV(sgd, parameters, verbose=1, n_jobs=1)
    # # clf = sgd
    # clf = KMeans(n_clusters=5, max_iter=10)
    # # clf = MultinomialNB()
    # # pdb.set_trace()
    # # clf.fit(vectorized, stars.map(str))

    # scores = cross_val_score(clf, vectorized, stars.map(str), cv=5, scoring='adjusted_rand_score')
    # print(scores)
    # # print(clf)

# ====================================================================================

    files = ['business.json', 'checkin.json', 'review.json', 'tip.json']#, 'user.json'] # 'photos.json'
    df = load.load_data(prefix + files[0])
    business = df.copy()
    # stars = business.stars
    # 
    for file in files[1:]:
        file_df = load.load_data(prefix + file)
        # Dropping stars since this is how we will judge our accuracy
        if file_df.get('stars') is not None:
            file_df.drop('stars', axis=1, inplace=True)
        df = df.merge(file_df, on='business_id', how='outer')

    df.fillna(0, inplace=True)
    stars = df.stars
    df.drop('stars', axis=1, inplace=True)
    print(stars.shape)
    print(df.shape)
    text_df = pd.DataFrame()
    text_df['text'] = []
    for column in df.columns:
        text_df.text = text_df.text.map(str) + ' ' + df[column].map(str)

    # at this point, have df of all information about a business. Now, lets start doing ML!
    parameters = {'alpha':[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1, 0.5, 1, 2]}
    multi = MultinomialNB()
    clf = GridSearchCV(multi, parameters, verbose=1, n_jobs=1)

    pipe = Pipeline([('tfid', TfidfVectorizer()),
                    ('clf', clf)])

    scores = cross_val_score(pipe, text_df.text, stars.map(str),
                             cv=None, n_jobs=-1)
    print(scores)
    pipe.fit(text_df.text, stars.map(str))

    return df




if __name__ == '__main__':
    main()