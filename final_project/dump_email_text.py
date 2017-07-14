""" Dump persisted texts from emails """
from sklearn.externals import joblib

def dump_email_text():
    """ Function to dump persisted texts from emails """
    email_text = joblib.load("email_text.pkl")
    with open("email_text.txt", "w") as f:
        f.writelines(email_text)
        f.close()
    return

if __name__ == "__main__":
    dump_email_text()
