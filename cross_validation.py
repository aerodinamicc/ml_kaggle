# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:34:29 2020

@author: shadow
"""


import pandas as pd

# Read the data
import os
os.chdir("D:/git/ml_kaggle/")
data = pd.read_csv('input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

"""
Then, we define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions.

While it's possible to do cross-validation without pipelines, it is quite difficult!
Using a pipeline will make the code remarkably straightforward.
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

# We obtain the cross-validation scores with the cross_val_score() function from scikit-learn.
# We set the number of folds with the cv parameter.

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

"""
The scoring parameter chooses a measure of model quality to report: 
    in this case, we chose negative mean absolute error (MAE). 
    The docs for scikit-learn show a list of options.

It is a little surprising that we specify negative MAE. 
Scikit-learn has a convention where all metrics are defined so a high number is better. Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.

We typically want a single measure of model quality to compare alternative models. 
So we take the average across experiments.
"""

print("Average MAE score (across experiments):")
print(scores.mean())
"""
Average MAE score (across experiments):
277707.3795913405
Conclusion
Using cross-validation yields a much better measure of model quality,
with the added benefit of cleaning up our code: note that we no longer need to keep track
of separate training and validation sets. So, especially for small datasets, it's a good improvement!
"""


def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=n_estimators,
                                                                  random_state=0))
                                 ])


    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    return scores.mean()

results = list(map(lambda x: get_score(x), list(range(50, 450, 50))))
results = dict(zip(list(range(50, 450, 50)), results))