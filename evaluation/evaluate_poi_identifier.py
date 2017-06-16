#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
print "Score: ", clf.score(x_test, y_test)
y_predict = clf.predict(x_test)
import numpy as np
print "True positives: ", sum(np.logical_and(y_predict==y_test, y_test==1))
from sklearn.metrics import precision_score, recall_score
precision_score(y_true=y_test, y_pred=y_predict)
recall_score(y_true=y_test, y_pred=y_predict)
