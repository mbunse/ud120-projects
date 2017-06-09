#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm

# Reduce size of training samples
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


# C_VALS = [10, 100, 1000, 10000]
C_VALS = [10000]
for c_val in C_VALS:

    clf = svm.SVC(kernel='rbf', C=c_val)
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"


    t1 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t1, 3), "s"

    from sklearn.metrics import accuracy_score
    print c_val, ": \t", accuracy_score(labels_test, pred)

    TEST_ITEMS = [10, 26, 50]
    for item in TEST_ITEMS:
        print item, ": \t", pred[item]

    print "# in class 1: ", (pred == 1).sum()
    # An alternative way would be
    # print (clf.score(features_test, labels_test))

# To save the fitted model to a file
# uncomment  the following lines of code
import pickle
pickle.dump(clf, open('svm_clf_mail','w'), 2)

#########################################################


