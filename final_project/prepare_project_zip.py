""" Helper module for project submission """
from zipfile import ZipFile
from os import system

def export_notebook():
    """ Function to export jupyter notebook as HTML file """
    system("jupyter nbconvert --to HTML \"Look At Enron data set.ipynb\"")
    return

def zip_files():
    """ Function to zip files for project
    submission """
    zipper = ZipFile("Moritz_Bunse_ML_project.zip", "w")
    files_to_write = ["poi_id.py",
                      "my_classifier.pkl",
                      "my_dataset.pkl",
                      "my_feature_list.pkl",
                      "tester.py",
                      "Look+At+Enron+data+set.html",
                      "Look At Enron data set.ipynb",
                      "data_dict.pkl",
                      "final_project_dataset.pkl",
                     ]
    for filename in files_to_write:
        zipper.write(filename)

    zipper.close()

if __name__ == "__main__":
    export_notebook()
    zip_files()

    TEXT = """I compiled a Jupyter notebook included in the zip file. The git repo can be found here: https://github.com/mbunse/ud120-projects

I had to modify the tester.py module to work with the data set. (As some feature selection steps are included in the classifier the feature vector has to be formatted as dictionary which is not supported by the original version of this module). I included an updated version of this module in the archive. """
    print TEXT
