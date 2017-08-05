#!/usr/bin/python

import sys
import os
import pickle
import string

from time import time
#from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split, cross_val_score,  \
    cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, SelectPercentile, \
    SelectKBest, f_classif
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler, FunctionTransformer, Imputer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from scipy.sparse import vstack as sparse_vstack

from nltk.stem.snowball import SnowballStemmer
import re
#from parse_out_email_text import parseOutText

FEATURES_FINANCIAL = ['salary',
                      #'deferral_payments',
                      'total_payments',
                      #'loan_advances',
                      'bonus',
                      #'restricted_stock_deferred',
                      #'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      #'director_fees'
                    ]
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
    def __init__(self, email_part="text", from_to="from"):
        if email_part not in ("text", "subject"):
            raise ValueError("email_part must be either 'text' or 'subject'")
        self.email_part = email_part
        if from_to not in ("from", "to"):
            raise ValueError("from_to must be either 'from' or 'to'")
        self.from_to = from_to

    def parseOutText(self, f):
        """ given an opened email file f, parse out all text below the
            metadata block at the top
            (in Part 2, you will also add stemming capabilities)
            and return a string that contains all the words
            in the email (space-separated) 
            
            example use case:
            f = open("email_file_name.txt", "r")
            text = parseOutText(f)
            
            """


        f.seek(0)  ### go back to beginning of file (annoying)
        all_text = f.read()
        ### split off metadata
        content = re.split("X-FileName:.*$", all_text, flags=re.MULTILINE, maxsplit=1)
        words = ""
        # Check if content has been found
        if len(content) > 1:
            text_string = content[1]
            ## remove mails that are forwarded or to which are responded
            # e.g. ---------------------- Forwarded"
            text_string = re.split(r"-*\sForwarded", text_string, maxsplit=1)[0]

            # -----Original Message-----
            text_string = re.split(r"-*\Original\sMessage", text_string, maxsplit=1)[0]

            # Vince J Kaminski@ECT
            # 04/30/2001 02:28 PM
            # To:	Stanley Horton/Corp/Enron@Enron, Danny McCarty/ET&S/Enron@Enron
            # cc:	Vince J Kaminski/HOU/ECT@ECT 
            # or
            # Vince J Kaminski@ECT
            # 04/30/2001 02:28 PM
            # to:	Stanley Horton/Corp/Enron@Enron, Danny McCarty/ET&S/Enron@Enron
            # cc:	Vince J Kaminski/HOU/ECT@ECT 
            text_string = re.split(r"((.*\n){2})[Tt]o:\s", text_string, maxsplit=1)[0]
            
            # remove punctuation
            text_string = text_string.translate(None, string.punctuation)

            ### split the text string into individual words, stem each word,
            ### and append the stemmed word to words (make sure there's a single
            ### space between each stemmed word)
            from nltk.stem.snowball import SnowballStemmer

            stemmer = SnowballStemmer("english")
            words = [stemmer.stem(word) for word in text_string.split()]

        return " ".join(words)

    def parse_out_subject(self, f):
        f.seek(0)  ### go back to beginning of file (annoying)
        all_text = f.read()
        ### split off metadata
        match = re.search(r"^Subject: ((Re:|Fwd:|Fw:)* *)*(?P<subject>.*)$", all_text, re.IGNORECASE | re.MULTILINE)
        if match.lastgroup == "subject":
            text_string = match.group("subject")
            text_string = text_string.translate(None, string.punctuation)
            return text_string
        return ""

    def fit(self, x, y=None):
        return self

    def transform(self, features):
        """ concatenate all emails texts """
        new_features = [] 
        set_ob_subjects = set()
        for item in features:
            email_address = item["email_address"]
            email_filename = "emails_by_address/" + self.from_to + "_" + email_address + ".txt"
            try:
                with open(email_filename, "r") as from_person:
                    email_text = ""
                    for path in from_person:
                        path = path.replace("enron_mail_20110402/", "")
                        path = os.path.join('..', path[:-1])
                        #print path
                        try:
                            email = open(path, "r")

                            ### Maybe use str.replace() to remove any instances of the words
                            ### use parseOutText to extract the text from the opened email
                            if self.email_part == "text":
                                email_text = " ".join([email_text, self.parseOutText(email)])
                            else:
                                email_text = ",".join([email_text, self.parse_out_subject(email)])
                            email.close()
                        except IOError:
                            print email + " not found"
                        ### append the text to word_data
                    #print "email_text"
                    #print email_text
                    new_features.append(email_text)
            except IOError:
                print email_filename + " not found"
                new_features.append("")
        return new_features

class DropSelectedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_match_keys = []):
        self.drop_match_keys = drop_match_keys

    def fit(self, x, y=None):
        self.drop_feature_keys = []
        for match_key in self.drop_match_keys:
            self.drop_feature_keys += filter(re.compile(match_key).match, x[0].keys())
        return self

    def transform(self, x):
        reduced_features = []
        i_item = 0
        for item in x:
            new_item = {}
            i_item += 1
            print "item: ", i_item
            for key, value in item.items():
                if key not in self.drop_feature_keys:
                    if value=="NaN":
                        value = np.nan
                    new_item[key] = np.float(value)
            reduced_features.append(new_item)
        return reduced_features

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_feature_names(self):
        return self.drop_feature_keys

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

class SelectFeatureList(BaseEstimator, TransformerMixin):
    def __init__(self, selected_feature_list, convert_to_numeric=False):
        self.selected_feature_list = selected_feature_list
        self.convert_to_numeric = convert_to_numeric

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        reduced_features = []
        for item in x:
            new_item = {}
            for key in self.selected_feature_list:
                if self.convert_to_numeric:
                    if item[key]=="NaN":
                        new_item[key] = np.nan
                    else:
                        new_item[key] = np.float(item[key])
                else:
                    new_item[key] = item[key]
            reduced_features.append(new_item)
        return reduced_features

class SelectMatchFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_match, convert_to_numeric=False):
        self.feature_match = feature_match
        self.match_keys = None
        self.convert_to_numeric = convert_to_numeric

    def fit(self, x, y=None):
        # calculate match keys from first element
        self.match_keys = filter(re.compile(self.feature_match).match, x[0].keys())
        return self

    def transform(self, x):
        reduced_features = []
        for item in x:
            sample_features = []
            for key in self.match_keys:
                if self.convert_to_numeric:
                    if item[key]=="NaN":
                        item[key] = np.nan
                    else:
                        item[key] = np.float(item[key])

                sample_features.append(item[key])
            reduced_features.append(sample_features)

        return np.asanyarray(reduced_features)

    def get_feature_names(self):
        return self.match_keys

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
            new_item["word_features_transformed"] = word_features_transformed[idx].toarray()
            new_features.append(new_item)
        return new_features

class KNeighborsTransformer(KNeighborsClassifier):
    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=1,
                 **kwargs):
        KNeighborsClassifier.__init__(self, n_neighbors,
                 weights, algorithm, leaf_size,
                 p, metric, metric_params, n_jobs,
                 **kwargs)

    def transform(self, x):
        pred = self.predict(x)
        new_features = []
        for item in pred:
            new_features.append([item])
        return new_features

    def fit_transform(self, x, y):
        self.fit(x, y)
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
            print "Model with rank: {0}".format(i)
            print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])
            print "Parameters: {0}".format(results['params'][candidate])
            print ""

def log_trans(x):
    """ log transformation as helper function 
    lambda function can be pickeled """
    return np.log1p(np.abs(x))

def build_poi_id_model(features, labels):

    # Split into training and testing
    splitter = StratifiedKFold(n_splits=10)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.05,
                         random_state=123456,
                         stratify=labels
                        )


    # Setting for persistance
    # In the persistance run, texts from emails are extracted for
    # training and testing data sets and the results are persisted
    # to files.
    # If not in persistance run, these files are only loaded and
    # processing of the emails is skipped

    # Pipeline to process email texts
    # First, extract texts from person
    # then, eventually persist those texts
    # then, vectorize the texts
    # then, select only the percentile with the most separating power
    # then, convert result to dense array (needed for some classifiers)
    pipeline_email_text = Pipeline([
        ("GetEmailText", SelectMatchFeatures(feature_match="word_.*")),
        #("SelectPercentile", SelectPercentile(score_func=f_classif, percentile=10)),
        ("SelectPercentile", SelectKBest(score_func=f_classif, k=20)),
        #("SVC", SelectFromModel(LinearSVC(class_weight="balanced", C=0.7), threshold=0.25)),
        ("NaiveBayes", SelectFromModel(MultinomialNB(alpha=1, fit_prior=False), threshold=0.5)),
        #("NaiveBayes", MultinomialNBTransformer(alpha=1, fit_prior=False)),
        #("Scale", StandardScaler()),
    ])

    pipeline_subjects = Pipeline([
        ("GetEmailText", SelectMatchFeatures(feature_match="sub_.*")),
        ("SelectPercentile", SelectKBest(score_func=f_classif, k=10)),
        #("NaiveBayes", SelectFromModel(MultinomialNB(alpha=1, fit_prior=False))),
        #("NaiveBayes", MultinomialNBTransformer(alpha=1, fit_prior=False)),
        #("Scale", StandardScaler())
    ])
    # Process financial features
    pipeline_financial = Pipeline([
        ("Selector", 
         SelectFeatureList(selected_feature_list=FEATURES_FINANCIAL, convert_to_numeric=True)),
        ("ConvertToVector", DictVectorizer(sparse=False)),
        ("Imputer", Imputer(strategy="median")),
        ("Log1P", FunctionTransformer(func=log_trans)),
    ])


    # Process other features
    # First, drop email_adress feature, which is only needed to
    # load the email texts
    # then, convert dictionary to dense vector
    pipeline_email = Pipeline([
        ("Selector", 
         SelectFeatureList(
             selected_feature_list=FEATURES_EMAIL, convert_to_numeric=True)),
        ("ConvertToVector", DictVectorizer(sparse=False)),
        ("Imputer", Imputer(strategy="median")),
        ("Log1P", FunctionTransformer(func=log_trans)),
    ])

    # Combine email text features and other features
    # then run classifier on these features
    pipeline_union = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                #("email_text", pipeline_email_text),
                ("subjects", pipeline_subjects),
                #("financial", pipeline_financial),
                ("email",pipeline_email),
            ]
        )),
        ("Scale", StandardScaler()),
        #("Select", SelectKBest(score_func=f_classif, k=10)),
        #("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        #("SVC", LinearSVC(class_weight="balanced")),
        #("SVC", SVC(C=1, kernel='rbf')),
        ("DecisionTree", RandomForestClassifier(n_estimators=20, min_samples_split=2, min_samples_leaf=1, class_weight=None)),
    ])

    # Fit the complete pipeline
    # Test accuracy of model
    # param_grid_union = {
    #     "DecisionTree__min_samples_split": [2,4,6],
    #     "DecisionTree__min_samples_leaf": [1,2,4],
    #     "DecisionTree__n_estimators": [5, 10, 20],
    #     }

    # grid_search_union = GridSearchCV(pipeline_union, param_grid=param_grid_union, cv=10)
    #start = time()
    #grid_search_union.fit(features, labels)

    #print("GridSearchCV took %.2f seconds for %d candidate parameter settings." 
    #    % (time() - start, len(grid_search_union.cv_results_['params'])))
    #report(grid_search_union.cv_results_)
    np.random.seed(42)
    pred = cross_val_predict(pipeline_union, features, labels, cv=10)
    print confusion_matrix(labels, pred)
    print classification_report(labels, pred)
    print "Accuracy: ", accuracy_score(labels, pred)

    # word_feature_names = SelectMatchFeatures(convert_to_numeric=True, feature_match="word_.*").fit(features).get_feature_names()
    sub_feature_names = SelectMatchFeatures(convert_to_numeric=True, feature_match="sub_.*").fit(features).get_feature_names()

    features_list = FEATURES_EMAIL + sub_feature_names

    return pipeline_union, features_list

def prepare_data(data_dict, filename="data_dict.pkl", load=True):
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

    # Extract email texts    
    emailtext_extractor = GetEmailText(email_part="text", from_to="from")
    email_texts = emailtext_extractor.transform(data_dict.values())

    # Vectorize texts
    text_vectorizer = TfidfVectorizer(max_df=0.3, min_df=1, stop_words='english', token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b", max_features=500)
    #text_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words='english', token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b")
    text_vect = text_vectorizer.fit_transform(email_texts)
    text_vect_words = text_vectorizer.get_feature_names()

    # Extract from subjects
    subject_from_extractor = GetEmailText(email_part="subject", from_to="from")
    subjects_from = subject_from_extractor.transform(data_dict.values())

    # Vectorize from subjects
    sub_vectorizer = TfidfVectorizer(max_df=0.3, min_df=1, token_pattern=r"[^,]+", max_features=100)
    #sub_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, token_pattern=r"[^,]+")
    subjects_from_vect = sub_vectorizer.fit_transform(subjects_from)
    selected_subs_from = sub_vectorizer.get_feature_names()

    #vectorizer = DictVectorizer(sparse=False)
    #subjects_from_vect = vectorizer.fit_transform(subjects_from)
    #selected_subs_from = vectorizer.get_feature_names()
    print "Selected subjects from:"
    print selected_subs_from[0:10]

    # Extract to subjects
    subject_to_extractor = GetEmailText(email_part="subject", from_to="to")
    subjects_to = subject_to_extractor.transform(data_dict.values())

    # Vectorize to subjects
    subjects_to_vect = sub_vectorizer.fit_transform(subjects_to)
    selected_subs_to = sub_vectorizer.get_feature_names()
    print "Selected subjects to:"
    print selected_subs_to[0:10]

    # Add text and subject features to data_dict
    for idx, value in enumerate(data_dict.values()):
        for idx_word, word in enumerate(text_vect_words):
            value["word_" + word] = text_vect[idx, idx_word]
        for idx_sub, subject in enumerate(selected_subs_from):
            value["sub_from_" + subject] = subjects_from_vect[idx, idx_sub]
        for idx_sub, subject in enumerate(selected_subs_to):
            value["sub_to_" + subject] = subjects_to_vect[idx, idx_sub]
        value.pop("email_address", None)

    # Save data to file
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

        # remove files without emails
        for key, value in data_dict.items():
            if value["to_messages"]=="NaN":
                data_dict.pop(key,None)

        # Remove data with all missing values
        for key, value in data_dict.items():
            has_values = False
            for subkey, subval in value.items():
                if subkey != "email_address" and subval=="NaN":
                    has_values = True
                    break
            if not has_values:
                data_dict.pop(key,None)

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
            feature = value.copy()
            feature.pop("poi",None)
            features.append(feature)
            names.append(key)
        clf, features_list = build_poi_id_model(features, labels)

        # Dump classifier for later 
        dump_classifier_and_data(clf, my_dataset, ["poi"] + features_list)
