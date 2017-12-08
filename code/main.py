#!/usr/bin/env python3
import pandas as pd
import numpy as np
import load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pdb

def main():
    prefix = '../playdata/'

    files = 'business.json'
    business_df = load.load_data(prefix + files)
    stars = business_df.stars
    business_df = business_df.drop('stars', axis=1)
    # base
    vect = TfidfVectorizer()
    
    df = pd.DataFrame()
    df['text'] = []
    for column in business_df.columns:
        df['text'] = df['text'].map(str) + ' ' + business_df[column].map(str)
    vectorized = vect.fit_transform(df.text)
    
    parameters = {'penalty': ('l2', 'elasticnet'), 'alpha': (0.00001, 0.000001)}
    sgd = SGDClassifier(max_iter=10)
    clf = GridSearchCV(sgd, parameters, verbose=1, n_jobs=-1)


   
    # pdb.set_trace()
    # clf.fit(vectorized, stars.map(str))

    scores = cross_val_score(clf, vectorized, stars.map(str), cv=5)
    print(scores)
    # files = ['business.json', 'checkin.json', 'review.json', 'tip.json']#, 'user.json'] # 'photos.json'
    # df = load.load_data(prefix + files[0])
    # business = df.copy()

    # for file in files[1:]:
    #     file_df = load.load_data(prefix + file)
    #     # Dropping stars since this is how we will judge our accuracy
    #     if file_df.get('stars') is not None:
    #         file_df.drop('stars', axis=1, inplace=True)
    #     df = df.merge(file_df, on='business_id', how='outer')

    # df.fillna(0, inplace=True)
    # at this point, have df of all information about a business. Now, lets start doing ML!

    # pipe = Pipeline([('vect', CountVectorizer()),
    #                  ('tfid', TfidfTransformer()),
    #                  ('clf', SGDClassifier())])

    # scores = cross_val_score(pipe, df.drop('stars', axis=1), df.stars,
    #                          cv=None, n_jobs=-1)
    # print(scores)
    # pip.fit(df.drop('stars', axis=1), df.stars)

    return df




if __name__ == '__main__':
    main()