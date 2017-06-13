#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import sys
import numpy
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def draw(pred, features, poi, mark_poi=False, name="image.png",
         f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for i_feature, prediction in enumerate(pred):
        plt.scatter(features[i_feature][0], features[i_feature][1], color=colors[prediction])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for i_feature, prediction in enumerate(pred):
            if poi[i_feature]:
                plt.scatter(features[i_feature][0], features[i_feature][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()






def run_cluster(features_list, data_dict):
    """
    Function to draw output from k-mean clustering algorithm.

    Parameters
    ----------
    features_list: list of features to use from second parameter
    
    data_dict: dictionary 

    Output
    ------
    No value is returned.
    """
    data = featureFormat(data_dict, features_list)
    poi, finance_features = targetFeatureSplit(data)


    ### in the "clustering with 3 features" part of the mini-project,
    ### you'll want to change this line to
    ### for f1, f2, _ in finance_features:
    ### (as it's currently written, the line below assumes 2 features)
    for features in finance_features:
        plt.scatter(features[0], features[1])
    plt.show()

    ### cluster here; create predictions of the cluster labels
    ### for the data and store them to a list called pred

    from sklearn import cluster

    pred = cluster.KMeans(n_clusters=2).fit_predict(finance_features)

    ### rename the "name" parameter when you change the number of features
    ### so that the figure gets saved to a different file
    try:
        draw(pred, finance_features, poi, mark_poi=False,
             name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
    except NameError:
        print "no predictions object named pred found, no clusters to plot"


def test_k_means():
    """
    Function to test k_means cluster
    """
    # load in the dict of dicts containing all the data on each person in the dataset
    data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
    # there's an outlier--remove it!
    data_dict.pop("TOTAL", 0)
    feature_1 = "salary"
    feature_2 = "exercised_stock_options"
    poi = "poi"
    features_list = [poi, feature_1, feature_2]
    # run_cluster(features_list)
    run_cluster(features_list + ["total_payments"], data_dict)

    exercised_stock_options = []
    salaries = []
    for key, value in data_dict.items():
        if value["exercised_stock_options"] <> "NaN":
            exercised_stock_options.append(float(value["exercised_stock_options"]))
        if value["salary"] <> "NaN":
            salaries.append(float(value["salary"]))
    print "min exercised_stock_options:", min(exercised_stock_options)
    print "max exercised_stock_options:", max(exercised_stock_options)
    print "min salary:", min(salaries)
    print "max salary:", max(salaries)

    # Transform

    from sklearn.preprocessing import MinMaxScaler
    opt_scaler = MinMaxScaler()
    opt_scaler.fit(numpy.resize(exercised_stock_options, (-1, 1)))
    trfmd_ex_stock_opt = opt_scaler.transform(exercised_stock_options)
    
    sal_scaler = MinMaxScaler()
    sal_scaler.fit(numpy.resize(salaries, (-1, 1)))
    trfmd_salaries = sal_scaler.transform(salaries)

    print "Scaled salary of 200,000: ", sal_scaler.transform(numpy.resize([200000], (1,1)))
    print "Scaled exercised stock options price of 1 Mio.: ", opt_scaler.transform(numpy.resize([1000000], (1,1)))

if __name__ == "__main__":
    test_k_means()
