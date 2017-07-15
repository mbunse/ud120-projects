#!/usr/bin/python

import sys
import os
import pickle
from time import time
#from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from scipy.sparse import vstack as sparse_vstack
from parse_out_email_text import parseOutText

FEATURES_FINANCIAL = ['salary',
                      'deferral_payments',
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'restricted_stock_deferred',
                      'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      'director_fees']
FEATURES_EMAIL = ['to_messages',
                  'from_poi_to_this_person',
                  'from_messages',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']
TARGET = 'poi'

class StripKeys(BaseEstimator, TransformerMixin):
    """ transfrom enron data to features """
    def __init__(self):
        pass
    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict.values()

class GetEmailText(BaseEstimator, TransformerMixin):
    """ extract email texts for person """
    def __init__(self, skip=False):
        self.skip = skip
    def fit(self, x, y=None):
        return self

    def transform(self, features):
        """ concatenate all emails texts """
        if self.skip:
            return features

        new_features = [] 
        for item in features:
            email_address = item["email_address"]
            try:
                with open("emails_by_address/from_" + email_address + ".txt", "r") as from_person:
                    email_text = ""
                    for path in from_person:
                        path = path.replace("enron_mail_20110402/", "")
                        path = os.path.join('..', path[:-1])
                        print path
                        email = open(path, "r")

                        ### Maybe use str.replace() to remove any instances of the words
                        ### use parseOutText to extract the text from the opened email
                        email_text = " ".join([email_text, parseOutText(email)])
                        email.close()
                    ### append the text to word_data
                new_features.append(email_text)
            except IOError:
                print "File not found"
                new_features.append("")
        return new_features

class DropSelectedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_feature_keys = []):
        self.drop_feature_keys = drop_feature_keys

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        reduced_features = []
        for item in x:
            new_item = {}
            for key, value in item.items():
                if key not in self.drop_feature_keys:
                    if value=="NaN":
                        value = 0
                    new_item[key] = float(value)
            reduced_features.append(new_item)
        return reduced_features

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, selected_feature):
        self.selected_feature = selected_feature

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        reduced_features = []
        for item in x:
            reduced_features.append(item[self.selected_feature])
        return reduced_features
    
class TfidfVectorizerForFeature(TfidfVectorizer):
    
    def __init__(self, word_data_key="email_text", **kwargs):
        self.word_data_key = word_data_key
        TfidfVectorizer.__init__(self, **kwargs)
    
    def transform(self, x):
        word_data = [item[self.word_data_key] for item in x]
        word_features = super(TfidfVectorizer, self).transform(word_data)
        new_features = []
        for idx, item in enumerate(x):
            new_item = {}
            for key, value in item:
                if key != self.word_data_key:
                    new_item[key] = value
            new_item["word_features"]=word_features[idx]
            new_features.append(new_item)
        return new_features

    def fit(self, x, y=None):
        fit_transform(self, x, y=None)
        return self 
    def fit_transform(self, x, y=None):
        word_data = [item[self.word_data_key] for item in x]
        word_features = super(TfidfVectorizer, self).fit_transform(word_data, y)
        new_features = []
        for idx, item in enumerate(x):
            item.pop(self.word_data_key, None)
            new_item = item
            new_item["word_features"]=word_features[idx]
            new_features.append(new_item)
        return new_features


class TfidfVectorizerDebug(TfidfVectorizer):
    
    def __init__(self, **kwargs):
        TfidfVectorizer.__init__(self, **kwargs)
    
    def transform(self, x):
        print "TfidfVectorizerDebug.transform"
        print len(x)
        return super(TfidfVectorizer, self).transform(x)

    def fit(self, x, y=None):
        print "TfidfVectorizerDebug.fit"
        print len(x)
        fit_transform(self, x, y=None)
        return self 
    def fit_transform(self, x, y=None):
        print "TfidfVectorizerDebug.fit_transform"
        print len(x)
        return super(TfidfVectorizer, self).fit_transform(x, y)

# TODO: Put select percentile and tfidvecor in one step?
# use SelectPercentile.get_support(indices=False) to get the indices
# from  TfidfVectorizer.get_feature_names()
class SelectPercentileForFeature(SelectPercentile):
    def __init__(self, word_features_key="word_features", **kwargs):
        self.word_features_key = word_features_key
        SelectPercentile.__init__(self, **kwargs)
    def fit(self, x, y=None):
        word_features = [item[self.word_features_key] for item in x]
        word_features = sparse_vstack(word_features)
        super(SelectPercentile, self).fit(word_features, y)
        return self 
    def transform(self, x):
        word_features = [item[self.word_features_key] for item in x]
        word_features = sparse_vstack(word_features)
        word_features_transformed = super(SelectPercentile, self).transform(word_features)
        new_features = []
        for idx, item in enumerate(x):
            item.pop(self.word_features_key, None)
            new_item = item
            new_item["word_features_transformed"]=word_features_transformed[idx].toarray()
            new_features.append(new_item)
        return new_features

class DenseTransformer(BaseEstimator, TransformerMixin):
    """ https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required """
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class PersistAndLoadVector(BaseEstimator, TransformerMixin):
    """ Transformer class to dump and/or load a vector in fitting """
    def __init__(self, filename=None, persist=False, load=False, ):
        self.persist = persist
        self.load = load
        if (load or persist) and filename == None:
            raise ValueError("No filename given but persist or load set to true.")
        else:
            self.filename = filename 

    def fit(self, x, y=None):
        if self.persist:
            print "Dumping vector"
            joblib.dump(x, self.filename)
        return self

    def fit_transform(self, x, y=None):
        if self.persist:
            print "Dumping vector"
            joblib.dump(x, self.filename)          
        if self.load:
            print "fit_transform: Loading vector"
            vec = joblib.load(self.filename)
            print len(vec)
            return vec
        else:
            return x

    def transform(self, x):
        if self.persist:
            print "Dumping vector"
            joblib.dump(x, self.filename)          
        if self.load:
            print "transform: Loading vector"
            vec = joblib.load(self.filename)
            print len(vec)
            return vec
        else:
            return x
    

class MultinomialNBTransformer(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        MultinomialNB.__init__(self, alpha, fit_prior, class_prior)
    
    def transform(self, x):
        pred = self.predict(x)
        new_features = []
        for item in pred:
            new_features.append([item])
        return new_features

    def fit_transform(self, x, y):
        self.fit(x,y)
        pred = self.transform(x)
        return pred


def report(results, n_top=3):
    """
    Utility function to report best scores
    http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def build_poi_id_model(features, labels, persist_run = False):

    # Split into training and testing
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.1, random_state=42)

    # Setting for persistance
    # In the persistance run, texts from emails are extracted for 
    # training and testing data sets and the results are persisted
    # to files.
    # If not in persistance run, these files are only loaded and
    # processing of the emails is skipped

    
    persist = False
    load = True
    if persist_run:
        persist = True

    # Pipeline to process email texts
    # First, extract texts from person
    # then, eventually persist those texts
    # then, vectorize the texts
    # then, select only the percentile with the most separating power
    # then, convert result to dense array (needed for some classifiers)
    pipeline_email_text = Pipeline([
        ("GetEmailText", SelectFeatures(selected_feature="email_text")),
        ("VectorizeMail", TfidfVectorizer(sublinear_tf=True, max_df=0.02,
                                stop_words='english', token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b")),
        ("SelectPercentile", SelectPercentile(score_func=f_classif, percentile=5)),
        ("NaiveBayes", MultinomialNBTransformer(alpha=1.)),
    ])
    
    # Process other features
    # First, drop email_adress feature, which is only needed to
    # load the email texts
    # then, convert dictionary to dense vector
    pipeline_other = Pipeline([
        ("DropEmailAddress", DropSelectedFeatures(drop_feature_keys=["email_address", "email_text"])),
        ("ConvertToVector", DictVectorizer(sparse=False)),
        ("Scale", StandardScaler())
    ])

    # Combine email text features and other features
    # then run classifier on these features
    pipeline_union = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                ("email_text", pipeline_email_text),
                ("rest", pipeline_other)
            ]
        )),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=7))
    ])

    # Fit the complete pipeline
    # Test accuracy of model
    param_grid = {
        "union__email_text__VectorizeMail__max_df": [0.02, 0.05, 0.1],
        "union__email_text__VectorizeMail__ngram_range": [(1,1), (1, 2)],
        "union__email_text__SelectPercentile__percentile": [1, 2, 5],
        "union__email_text__NaiveBayes__alpha": [1, 0.8, 0.5],
        "KNeighborsClassifier__n_neighbors": [3, 5, 10],
        }

    grid_search = GridSearchCV(pipeline_union, param_grid=param_grid)
    start = time()
    grid_search.fit(features, labels)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." 
        % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)

    #scores = cross_val_score(pipeline_union, features, labels, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Dump word features selected by email text pipeline
    #selected_indices = pipeline_union.named_steps["union"].transformer_list[0][1].named_steps["SelectPercentile"].get_support(indices=True)
    #print np.take(pipeline_union.named_steps["union"].transformer_list[0][1].named_steps["VectorizeMail"].get_feature_names(),selected_indices)

    # Try classification only based on email texts
    pipeline_email_text_clf = Pipeline([
        ("GetEmailText", SelectFeatures(selected_feature="email_text")),
        ("VectorizeMail", TfidfVectorizerDebug(sublinear_tf=True, max_df=0.02,
                                               stop_words='english', token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b")),
        ("SelectPercentile", SelectPercentile(score_func=f_classif, percentile=5)),
        ("NaiveBayes", MultinomialNB())
    ])


    #scores = cross_val_score(pipeline_email_text_clf, features, labels, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Score is 0.8 for MultinomialNB, max_df=0.5, percentile=1
    # print "Score for only email text clf: ", pipeline_email_text_clf.score(features_test, labels_test)

    return

def prepare_data(data_dict, filename="data_dict.pkl", load=False):
    """ 
    If load is false, function takes basic input data, adds features,
    persists the final data and returns the data.
    
    Parameters
    ----------
    data_dict: a dictionary with key of sample id
               and value of dictionarys with key of 
               feature name and value of feature value
    filename:  filename for persiting or loading
    load:      if true, final data is just loaded.
               if false, data is transformed and persisted 
    """

    if load:
        return joblib.load(filename)
    
    emailtext_extractor = GetEmailText(skip=False)

    email_texts = emailtext_extractor.transform(data_dict.values())
    for idx, value in enumerate(data_dict.values()):
        value["email_text"] = email_texts[idx]
        value.pop("email_address", None)

    joblib.dump(data_dict, filename)
    return data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# was features_list = ['poi','salary'] # You will need to use more features

### Task 2: Remove outliers

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys=False, remove_all_zeroes=False)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ =="__main__":
    ### Load the dictionary containing the dataset

    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

        # drop TOTAL dataset:
        data_dict.pop("TOTAL",None)

        # Set to False to run on full sample
        if False:
            sample_to_use = dict(data_dict.items()[0:20])
        else:
            sample_to_use = data_dict

        my_dataset = prepare_data(sample_to_use, load=True)
        #print my_dataset
        # Split label
        features = []
        labels = []
        names = []
        for key, value in my_dataset.items():
            labels.append(value["poi"])
            value.pop("poi",None)
            features.append(value)
            names.append(key)
        build_poi_id_model(features, labels, persist_run = False)
