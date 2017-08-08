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
    SelectKBest, f_classif, chi2
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
    """ Transformator class to extract email texts for person

    Parameters
    ----------
    email_part: either "text" or "subject" indicating whether
                word from emial body or subject lines should be ectracted
    from_to:    either "from" or "to" indicating whether email sent
                or received by person.


    """
    def __init__(self, email_part="text", from_to="from"):
        if email_part not in ("text", "subject"):
            raise ValueError("email_part must be either 'text' or 'subject'")
        self.email_part = email_part
        if from_to not in ("from", "to"):
            raise ValueError("from_to must be either 'from' or 'to'")
        self.from_to = from_to

    def parseOutText(self, f):
        """
        given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        Parameters
        ----------
        f:  file object

        Return value
        ------------
        string with space separated words from email text with
        quoted emails removed.
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
        """ Function to extraxt subject line from a file object
        Parameters
        ----------
        f:  file object

        Return value
        ------------
        String with normalized subject line
        """
        f.seek(0)  ### go back to beginning of file (annoying)
        all_text = f.read()
        ### split off metadata
        match = re.search(r"^Subject: ((Re:|Fwd:|Fw:)* *)*(?P<subject>.*)$",
                          all_text, re.IGNORECASE | re.MULTILINE)
        if match.lastgroup == "subject":
            text_string = match.group("subject")
            text_string = text_string.translate(None, string.punctuation)
            return text_string
        return ""

    def fit(self, x, y=None):
        return self

    def transform(self, features):
        """ concatenate all emails texts
        Parameters
        ----------
        features:   list of dictionaries for each person. Email adress is
                    taken from key "email_address". A text file in
                    sub-directory emails_by_address woth convention
                    <from/to>_<email_address>.txt is searched. The files
                    listed in this file are opened and processed.

        Return value
        ------------
        list of strings containing either space separated words or
        comma separated subject lines for given person
        """
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
    """ Transformator to drop keys from dictioniaries in a list
    matching a regular expression from a list. All keys have to
    be present in the first element of the feature list.

    Parameters
    ----------
    drop_match_keys:    list of regular expression for keys to be dropped
                        from dictionaries
    """
    def __init__(self, drop_match_keys = []):
        self.drop_match_keys = drop_match_keys

    def fit(self, x, y=None):
        """ find names of keys to be dropped from regex given
        by looking at the first dictionary in x

        Parameters
        ----------
        x:  list of dictionaries
        y:  not used

        Return value
        ------------
        Instance with found key names to be dropped
        """
        self.drop_feature_keys = []
        for match_key in self.drop_match_keys:
            self.drop_feature_keys += filter(re.compile(match_key).match, x[0].keys())
        return self

    def transform(self, x):
        """
        Function to drop keys from dictionaries in list

        Parameters
        ----------
        x:  list of dictionaries

        Return values
        -------------
        list of dictionaries with selected keys removed
        """
        reduced_features = []
        i_item = 0
        for item in x:
            new_item = {}
            i_item += 1
            print "item: ", i_item
            for key, value in item.items():
                if key not in self.drop_feature_keys:
                    if value == "NaN":
                        value = np.nan
                    new_item[key] = np.float(value)
            reduced_features.append(new_item)
        return reduced_features

    def fit_transform(self, x, y=None):
        """
        Fit and transform in one step. See `fit` and `transform`
        functions for more information.
        """

        self.fit(x, y)
        return self.transform(x)

    def get_feature_names(self):
        """ Returns names of dropped features """
        return self.drop_feature_keys

class SelectFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer class to select single value for given
    key from dictionaries in a list

    Parameters
    ----------
    selected_feature:   feature to extract from dictionaries
    """

    def __init__(self, selected_feature):
        self.selected_feature = selected_feature

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """
        Transfrom list of dictioniaries to list of single
        selected feature.

        Parameters
        ----------
        x:  list of dictionaries

        Return value
        ------------
        list of values
        """
        reduced_features = []
        for item in x:
            reduced_features.append(item[self.selected_feature])
        return reduced_features

class SelectFeatureList(BaseEstimator, TransformerMixin):
    """
    Transformer class to select list of features value for given
    keys from dictionaries in a list

    Parameters
    ----------
    selected_feature_list:   list of keys to extract from dictionaries
    convert_to_numeric: boolean, indicating whether features should be
                        casted to float. In this case, "NaN" is
                        translated to np.nan.
    """
    def __init__(self, selected_feature_list, convert_to_numeric=False):
        self.selected_feature_list = selected_feature_list
        self.convert_to_numeric = convert_to_numeric

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """
        Extract selected keys and transform to float if `convert_to_numeric`
        is True.

        Paramters
        ---------
        x:  list of dictionaries

        Return value
        ------------
        list of dictionaries with only the selected keys.
        """
        reduced_features = []
        for item in x:
            new_item = {}
            for key in self.selected_feature_list:
                if self.convert_to_numeric:
                    if item[key] == "NaN":
                        new_item[key] = np.nan
                    else:
                        new_item[key] = np.float(item[key])
                else:
                    new_item[key] = item[key]
            reduced_features.append(new_item)
        return reduced_features

class SelectMatchFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer class to select list of features value for
    keys matching a regular expression from dictionaries in a list.
    Regular expression matching is only applied to first dictionary
    in list of dictionaries.

    Parameters
    ----------
    feature_match:      regular expression for keys to extract from
                        dictionaries
    convert_to_numeric: boolean, indicating whether features should be
                        casted to float. In this case, "NaN" is
                        translated to np.nan.
    """
    def __init__(self, feature_match, convert_to_numeric=False):
        self.feature_match = feature_match
        self.match_keys = None
        self.convert_to_numeric = convert_to_numeric

    def fit(self, x, y=None):
        # calculate match keys from first element
        self.match_keys = filter(re.compile(self.feature_match).match, x[0].keys())
        return self

    def transform(self, x):
        """
        Extract selected matching keys and transform to float if `convert_to_numeric`
        is True.

        Paramters
        ---------
        x:  list of dictionaries

        Return value
        ------------
        list of list with only the selected keys.
        """
        reduced_features = []
        for item in x:
            sample_features = []
            for key in self.match_keys:
                if self.convert_to_numeric:
                    if item[key] == "NaN":
                        item[key] = np.nan
                    else:
                        item[key] = np.float(item[key])

                sample_features.append(item[key])
            reduced_features.append(sample_features)

        return np.asanyarray(reduced_features)

    def get_feature_names(self):
        """ Return selected keys """
        return self.match_keys

class ImputeOrZero(Imputer):
    """
    Transformer class to impute NaN values or set them to zero.

    Parameters
    ----------
    missing_values: integer or "NaN", optional (default="NaN")
                    see `Imputer` documentation
    strategy:   String, one of ["mean", "median", "most_frequent", "zero"]
    axis:       integer, optional (default=0)
                see `Imputer` documentation  
    verbose:    integer, optional (default=0)
                see `Imputer` documentation
    copy:       boolean, optional (default=True)
                see `Imputer` documentation
    """
    def __init__(self, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        if strategy not in ["mean", "median", "most_frequent", "zero"]:
            raise ValueError("strategy has be one of the following values: " +
                "\"mean\", \"median\", \"most_frequent\", \"zero\"")
        self.strategy = strategy
        if self.strategy == "zero":
            super(ImputeOrZero, self).__init__(missing_values=missing_values,
                             strategy="mean", axis=axis, verbose=verbose, 
                             copy=copy)
        else:
            super(ImputeOrZero, self).__init__(missing_values=missing_values,
                             strategy=strategy, axis=axis, verbose=verbose,
                             copy=copy)

    def fit(self, X, y):
        """ function to fit imputer """
        if self.strategy == "zero": 
            return self
        else:
            return super(ImputeOrZero, self).fit(X, y)
    
    def transform(self, X):
        if self.strategy == "zero":
            transformed_datasets = []
            for dataset in X:
                transformed_features = []
                for feature in dataset:
                    if np.isnan(feature):
                        transformed_features.append(0.)
                    else:
                        transformed_features.append(feature)
                transformed_datasets.append(transformed_features)
            return np.array(transformed_datasets)
        else:
            return super(ImputeOrZero, self).transform(X)
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

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
    """
    Function to train classifier to predict labels given features

    Parameters
    ----------
    features:   list of dictionaries per dataset
    lables:     list of boolean labels

    Return values
    clf, features_list
    clf:            trained classifier
    features_list:  list of features used by the classifier
    """
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
        ("SelectPercentile", SelectKBest(score_func=chi2, k=250)),
        #("SVC", SelectFromModel(LinearSVC(class_weight="balanced", C=0.7), threshold=0.25)),
        # ("NaiveBayes", SelectFromModel(MultinomialNB(alpha=.5, fit_prior=False), threshold=0.5)),
        #("Scale", StandardScaler()),
    ])

    pipeline_subjects = Pipeline([
        ("GetEmailText", SelectMatchFeatures(feature_match="sub_.*")),
        ("SelectPercentile", SelectKBest(score_func=chi2, k=100)),
        # ("NaiveBayes", SelectFromModel(MultinomialNB(alpha=1, fit_prior=False))),
        #("Scale", StandardScaler())
    ])
    # Process financial features
    pipeline_financial = Pipeline([
        ("Selector", 
         SelectFeatureList(selected_feature_list=FEATURES_FINANCIAL, convert_to_numeric=True)),
        ("ConvertToVector", DictVectorizer(sparse=False)),
        ("Impute", ImputeOrZero(strategy="zero")),
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
        ("Log1P", FunctionTransformer(func=log_trans)),
    ])

    # Combine email text features and other features
    # then run classifier on these features
    pipeline_union = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                ("email_text", pipeline_email_text),
                ("subjects", pipeline_subjects),
                ("financial", pipeline_financial),
                ("email",pipeline_email),
            ],
            #transformer_weights={'email_text': 0, 'subjects': 1, 'financial': 1, 'email': 1},
        )),
        ("Scale", StandardScaler()),
        #("Select", SelectKBest(score_func=f_classif, k=10)),
        # ("KNeighborsClassifier", KNeighborsClassifier()),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=1, metric='minkowski', weights='distance')),
        # ("SVC", SVC(class_weight='balanced')),
        #  ("SVC", SVC(C=0.8, kernel='rbf', class_weight='balanced')),
        #  ("DecisionTree", RandomForestClassifier()),
        #("DecisionTree", RandomForestClassifier(n_estimators=10, min_samples_split=6, min_samples_leaf=1, class_weight=None)),
        #("NaiveBayes", MultinomialNB(alpha=1, fit_prior=False)),
    ])

    # Fit the complete pipeline
    # Test accuracy of model
    param_grid_union = {
        "union__transformer_weights": [
                                    #    {'email_text': 1, 'subjects': 1, 'financial': 1, 'email': 1},
                                    #    {'email_text': 0, 'subjects': 1, 'financial': 1, 'email': 1},
                                    #    {'email_text': 1, 'subjects': 0, 'financial': 1, 'email': 1},
                                    #    {'email_text': 1, 'subjects': 1, 'financial': 0, 'email': 1},
                                    #    {'email_text': 1, 'subjects': 1, 'financial': 1, 'email': 0},
                                    #    {'email_text': 0, 'subjects': 0, 'financial': 1, 'email': 1},
                                       {'email_text': 0, 'subjects': 1, 'financial': 0, 'email': 1},
                                    #    {'email_text': 0, 'subjects': 1, 'financial': 1, 'email': 0},
                                    #    {'email_text': 1, 'subjects': 0, 'financial': 0, 'email': 1},
                                    #    {'email_text': 1, 'subjects': 0, 'financial': 1, 'email': 0},
                                    #    {'email_text': 1, 'subjects': 1, 'financial': 0, 'email': 0},
                                    #    {'email_text': 0, 'subjects': 0, 'financial': 0, 'email': 1},
                                    #    {'email_text': 0, 'subjects': 0, 'financial': 1, 'email': 0},
                                    #    {'email_text': 0, 'subjects': 1, 'financial': 0, 'email': 0},
                                    #    {'email_text': 1, 'subjects': 0, 'financial': 0, 'email': 0},
                                      ],
        # "union__email_text__SelectPercentile__k": [10, 50, 100, 250, 500],
        # "union__email_text__SelectPercentile__score_func": [chi2, f_classif],
        # "union__subjects__SelectPercentile__k": [2, 3, 5, 10, 100, 200],
        # "union__subjects__SelectPercentile__score_func": [chi2, f_classif],
        # "union__financial__Impute__strategy": ["median", "zero"],
        # "DecisionTree__min_samples_split": [2,4,6],
        # "DecisionTree__min_samples_leaf": [1,2,4],
        # "DecisionTree__n_estimators": [5, 10, 20],
        # "NaiveBayes__alpha": [.5, .8, 1],
        #  "SVC__C": [.2, .5, .8, 1],
        #  "SVC__kernel": ['rbf', 'sigmoid', 'linear'],
        #  "SVC__class_weight": [None, 'balanced'],
        #  "SVC__probability": [False, True],
        "KNeighborsClassifier__n_neighbors": [1, 3, 5],
        "KNeighborsClassifier__weights": ["uniform", "distance"],
        "KNeighborsClassifier__metric": ["minkowski", "manhattan"]
        }

    grid_search_union = GridSearchCV(pipeline_union, param_grid=param_grid_union, cv=10, scoring="f1")
    start = time()
    np.random.seed(42)
    grid_search_union.fit(features, labels)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." 
        % (time() - start, len(grid_search_union.cv_results_['params'])))
    report(grid_search_union.cv_results_)


    np.random.seed(42)
    best_est = np.flatnonzero(grid_search_union.cv_results_['rank_test_score'] == 1)[0]
    print grid_search_union.cv_results_['params'][best_est]
    pipeline_union.set_params(**grid_search_union.cv_results_['params'][best_est])

    pred = cross_val_predict(pipeline_union, features, labels, cv=10)
    print confusion_matrix(labels, pred)
    print classification_report(labels, pred)
    print "Accuracy: ", accuracy_score(labels, pred)

    # Extract names of subject features
    sub_feature_names = SelectMatchFeatures(convert_to_numeric=True, feature_match="sub_.*").fit(features).get_feature_names()
    features_list = sub_feature_names

    # Return classifier and names of features used
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

def extract_labels_features(data, label_key = "poi"):
    """ Function extracts labels and features arrays
    for classifier from data

    Parameters
    ----------
    data: dictionary of dictonaries including the key 
          `label_key` defining the label of each dataset

    Output
    ------
    labels, features: labels is a list of labels per data set,
                      feature is a list of dictionaries with features
                      for each data set.
    """
    features = []
    labels = []
    names = []
    for key, value in data.items():
        labels.append(value["poi"])
        feature = value.copy()
        feature.pop("poi",None)
        features.append(feature)
        names.append(key)
    
    return labels, features, names

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
        labels, features, _ = extract_labels_features(my_dataset)

        clf, features_list = build_poi_id_model(features, labels)

        # Dump classifier for later 
        dump_classifier_and_data(clf, my_dataset, ["poi"] + features_list)
