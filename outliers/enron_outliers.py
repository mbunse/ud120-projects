#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop( "TOTAL", 0 )
data = featureFormat(data_dict, features)

### your code below
# Find key for outlier
# for key, value in data_dict.items():
#     if value["salary"] <> "NaN" and value["salary"] > 23000000:
#         print value["salary"], key

for key, value in data_dict.items():
    if value["salary"] <> "NaN" and value["salary"] > 1000000 and \
        value["bonus"] <> "NaN" and value["bonus"] > 5000000:
        print value["bonus"], value["salary"], key

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


