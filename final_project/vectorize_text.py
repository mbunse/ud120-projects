#!/usr/bin/python

import os
import pickle
import re
import sys


from parse_out_email_text import parseOutText

sys.path.append("../tools")
from feature_format import featureFormat
def vectorize_email_text(email_addresses, poi_flags):
    """
        Starter code to process the emails from Sara and Chris to extract
        the features and get the documents ready for classification.

        The list of all the emails from Sara are in the from_sara list
        likewise for emails from Chris (from_chris)

        The actual documents are in the Enron email dataset, which
        you downloaded/unpacked in Part 0 of the first mini-project. If you have
        not obtained the Enron email corpus, run startup.py in the tools folder.

        The data is stored in lists and packed away in pickle files at the end.
    """

    from_data = []
    word_data = []

    ### temp_counter is a way to speed up the development--there are
    ### thousands of emails from Sara and Chris, so running over all of them
    ### can take a long time
    ### temp_counter helps you only look at the first 200 emails in the list so you
    ### can iterate your modifications quicker
    temp_counter = 0


    for poi_flag, email_address in zip(poi_flags, email_addresses):
        try:
            with open("emails_by_address/from_" + email_address + ".txt", "r") as from_person:
                email_text = ""
                for path in from_person:
                    ### only look at first 200 emails when developing
                    ### once everything is working, remove this line to run over full dataset
                    #temp_counter += 1
                    if temp_counter < 200:
                        path = path.replace("enron_mail_20110402/", "")
                        path = os.path.join('..', path[:-1])
                        print path
                        email = open(path, "r")

                        ### Maybe use str.replace() to remove any instances of the words

                        ### use parseOutText to extract the text from the opened email
                        email_text = " ".join([email_text, parseOutText(email)])
                        email.close()
                ### append the text to word_data
                word_data.append(email_text)
                from_data.append(poi_flag)
                

            print "emails from " + email_address + " processed"
        except IOError:
            pass
    pickle.dump(word_data, open("word_data.pkl", "w"))
    pickle.dump(from_data, open("email_authors.pkl", "w"))

    # word_data = pickle.load(open("your_word_data.pkl", "r"))
    # from_data =  pickle.load(open("your_email_authors.pkl", "r"))

    ### in Part 4, do TfIdf vectorization here
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_df=0.5)
    tfidf_word_data = vectorizer.fit_transform(word_data)

    print "Number of words: ", len(vectorizer.get_feature_names())
    print "Word number : ", vectorizer.get_feature_names()

    pickle.dump( tfidf_word_data, open("your_tfidf_word_data.pkl", "w") )

    # tfidf_word_data = pickle.load(open("your_tfidf_word_data.pkl", "r"))

def test():
    #vectorize_email_text(["sara.shackleton@enron.com", "chris.germany@enron.com"], [0, 0])
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        data = [[value["email_address"], value["poi"]] for key, value in data_dict.items() 
                 if value["email_address"] != "NaN"]
        print data
        email_addresses = []
        poi_flags = []
        for pair in data:
            email_addresses.append(pair[0])
            poi_flags.append(pair[1])
        vectorize_email_text(email_addresses, poi_flags)
if __name__ == "__main__":
    test()