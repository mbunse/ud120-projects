#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import re


def parseOutText(f):
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
    if len(content) > 1:
        text_string = content[1]

        ## remove mails that are forwarded or to which are responded
        # e.g. ---------------------- Forwarded"
        text_string = re.split("-*\sForwarded", text_string, maxsplit=1)[0]

        # -----Original Message-----
        text_string = re.split("-*\Original\sMessage", text_string, maxsplit=1)[0]

        # Vince J Kaminski@ECT
        # 04/30/2001 02:28 PM
        # To:	Stanley Horton/Corp/Enron@Enron, Danny McCarty/ET&S/Enron@Enron
        # cc:	Vince J Kaminski/HOU/ECT@ECT 
        # or
        # Vince J Kaminski@ECT
        # 04/30/2001 02:28 PM
        # to:	Stanley Horton/Corp/Enron@Enron, Danny McCarty/ET&S/Enron@Enron
        # cc:	Vince J Kaminski/HOU/ECT@ECT 
        
        text_string = re.split("((.*\n){2})[Tt]o:\s", text_string, maxsplit=1)[0]

        ### remove punctuation
        # should be autopmatically by scikit learn
        #text_string = text_string.translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in text_string.split()]



    return " ".join(words)

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text

    ff = open("..\maildir/jones-t/deleted_items/45", "r")
    text = parseOutText(ff)
    print text
    



if __name__ == '__main__':
    main()

