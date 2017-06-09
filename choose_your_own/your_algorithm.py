#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier

SETTING_N_NEIGHBORS = [2, 5, 10, 15, 30]
SETTING_WEIGHTS = ["uniform", "distance"]
SETTINGS_ALGORITHM = {"auto", "ball_tree", "kd_tree", "brute"}


for n_neighbors in SETTING_N_NEIGHBORS:
    for weights in SETTING_WEIGHTS:
        for algorithm in SETTINGS_ALGORITHM:
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                weights=weights
            )
            clf.fit(features_train, labels_train)

            print "n_neighbors: \t", n_neighbors
            print "weights: \t", weights
            print "algorithm: \t", algorithm
            print "accuray: ", clf.score(features_test, labels_test)




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
