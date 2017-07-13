#!/usr/bin/python

import sys
import os
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
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
    def __init__(self):
        pass
    def fit(self, x, y=None):
        return self

    def transform(self, features):
        """ concatenate all emails texts """
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
                new_features.append("")
        return new_features

class DropSelectedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_feature_keys):
        self.drop_feature_keys = drop_feature_keys

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        reduced_features = []
        for item in x:
            new_item = {}
            for key, value in item:
                if key not in drop_feature_keys:
                    new_item[key] = value
            reduced_features.append(new_item)
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

def build_poi_id_model(features, labels):
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    pipeline_email_text = Pipeline([
        ("GetEmailText", GetEmailText()),
        ("VectorizeMail", TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                stop_words='english', token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b")),
        ("SelectPercentile", SelectPercentile(score_func=f_classif, percentile=1))])
    
    # TODO: add sklearn.feature_extraction.DictVectorizer to pipeline

    # Test email pipeline returning 
    print pipeline_email_text.fit_transform(features, labels)
    selected_indices = pipeline_email_text.named_steps["SelectPercentile"].get_support(indices=True)
    print np.take(pipeline_email_text.named_steps["VectorizeMail"].get_feature_names(),selected_indices)
    return

def blub():
    features_list = [target] + features_financial + features_emails

    # was features_list = ['poi','salary'] # You will need to use more features

    ### Task 2: Remove outliers
    # for key, value in data_dict.items():
    #     if value["salary"] <> "NaN" and value["salary"] > 1000000 and \
    #         value["bonus"] <> "NaN" and value["bonus"] > 5000000:
    #         del data_dict[key]

    ### Task 3: Create new feature(s)
    words_file_handler = open("word_data.pkl", "r")
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    email_authors_handler = open("email_authors.pkl", "r")
    email_authors = pickle.load(email_authors_handler)
    email_authors_handler.close()
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys=False, remove_all_zeroes=False)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
    word_data_train, word_data_test, _, _ = \
        cross_validation.train_test_split(word_data, labels, test_size=0.1, random_state=42)

    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    word_data_train_transformed = vectorizer.fit_transform(word_data_train)
    word_data_test_transformed = vectorizer.transform(word_data_test)

    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(word_data_train_transformed, labels_train)
    word_data_train_transformed = selector.transform(word_data_train_transformed).toarray()
    word_data_test_transformed  = selector.transform(word_data_test_transformed).toarray()



    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # Provided to give you a starting point. Try a variety of classifiers.
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!

    clf.fit(np.hstack((features_train, word_data_train_transformed)), labels_train)

    print clf.score(np.hstack((features_test, word_data_test_transformed)), labels_test)

    clf.fit(word_data_train_transformed, labels_train)

    print clf.score(word_data_test_transformed, labels_test)
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ =="__main__":
    ### Load the dictionary containing the dataset

    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        test_sample = dict(data_dict.items()[0:20])

        # Split label
        features = []
        labels = []
        names = []
        for key, value in test_sample.items():
            labels.append(value["poi"])
            value.pop("poi",None)
            features.append(value)
            names.append(key)
        build_poi_id_model(features, labels)
