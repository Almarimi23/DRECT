import pandas as pd 
import numpy as np
import nltk
import re

# Repalce unwanted characters & words from challenge description
def replace_fn(df):
    col = 'challenge description'
    replace_chars = ['[', ']','*','**','(',')','***','Challenge Overview',
    'Challenge Objectives','Gigs Listing','Gigs Detail','Gigs Apply Page',
    'Technology Stack','Assets:']
    for char in replace_chars:
        df[col] = [df[col].loc[i].replace(char,' ') for i in range(len(df))]  
    return df

# Clean the text for descriptions
def utils_preprocess_desc(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   characters and then strip)
    text = re.sub(r'[^A-Za-z0-9(),!.?\'\`]', ' ', str(text).lower().strip())
    # Remove urls 
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[0-9\.]+', ' ', text)
    text = re.sub(r'[,\'"#@-_]+', ' ', text)
    text = ' '.join([w for w in text.split() if len(w) > 1])
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

# Clean the text for skills
def utils_preprocess_skills(text):
    ## clean (convert to lowercase and remove punctuations and   characters and then strip)
    text = re.sub(r'[^A-Za-z0-9(),!.?\'\`]', ' ', str(text).lower().strip())
    text = re.sub(r'[0-9\.]+', ' ', text)
    text = re.sub(r'[,\'"#@-_]+', ' ', text)
    text = ' '.join([w for w in text.split() if len(w) > 1])
    return text