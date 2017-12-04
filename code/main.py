#!/usr/bin/env python3
import pandas as pd
import load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.linear_model import SGDClassifier

def main():
    prefix = '../playdata/'
    files = ['business.json', 'checkin.json', 'review.json', 'tip.json']#, 'user.json'] # 'photos.json'
    df = load.load_data(prefix + files[0])
    business = df.copy()

    for file in files[1:]:
        file_df = load.load_data(prefix + file)
        # Dropping stars since this is how we will judge our accuracy
        if file_df.get('stars') is not None:
            file_df.drop('stars', axis=1, inplace=True)
        df = df.merge(file_df, on='business_id', how='outer')

    df.fillna(0, inplace=True)
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