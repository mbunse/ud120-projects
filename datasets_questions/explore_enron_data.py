#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
n_people = len(enron_data)
print "Persons: ", len(n_people)
print "Features per person: ", len(enron_data.itervalues().next())

n_poi = 0
for key, value in enron_data.items():
    if value['poi'] == 1:
        n_poi += 1
print "N POI: ", n_poi

for key, value in enron_data.items():
    if "PRENTICE" in key:
        print "Total stock value for ", key, ": ", value["total_stock_value"]

for key, value in enron_data.items():
    if "COLWELL" in key:
        print "Emails to POI from ", key, ": ", value["from_this_person_to_poi"]

for key, value in enron_data.items():
    if "SKILLING" in key:
        print "Exercised stock options from ", key, ": ", value["exercised_stock_options"]

for key, value in enron_data.items():
    if key in ["SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S"]:
        print "Total payment for ", key, ": ", value["total_payments"]

n_salary = 0
n_email_address = 0

for key, value in enron_data.items():
    if value['email_address'] <> "NaN":
        n_email_address += 1
    if value['salary'] <> "NaN":
        n_salary += 1
print "N email addresses: ", n_email_address
print "N salaries: ", n_salary

n_missing_total_pay = 0
for key, value in enron_data.items():
    if value['total_payments'] == "NaN":
        n_missing_total_pay += 1

print "N mssing total_payments: ", n_missing_total_pay
print "Perc. missing total payments: ", n_missing_total_pay/float(len(enron_data))

n_missing_total_pay_poi = 0
for key, value in enron_data.items():
    if value['poi'] == True and value['total_payments'] == "NaN":
        n_missing_total_pay_poi += 1

print "N missing total_payments POI: ", n_missing_total_pay_poi
print "Perc. missing total payments POI: ", n_missing_total_pay_poi/float(n_poi)
