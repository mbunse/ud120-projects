#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    from math import ceil
    cleaned_data = []
    ### your code goes here
    list_with_errors = []
    
    for pred, age, net_worth in zip(predictions, ages, net_worths):
        cleaned_data.append((age, net_worth, (pred-net_worth)**2))
    
    cleaned_data = sorted(cleaned_data, key=lambda s: s[2])
    cleaned_data = cleaned_data[0:int(len(predictions)*0.9)]

    return cleaned_data

