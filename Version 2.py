#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Dec 11 16:53:25 2018

@author: BERRIMI Mohamed  , Guelliani Sliman Nedjm Eddin 

VERSION 2 
"""

import nltk, string, numpy
#nltk.download('punkt') # first-time use only
stemmer = nltk.stem.porter.PorterStemmer()
def StemTokens(tokens):
     return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
     return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


 # nltk.download('wordnet') # first-time use only
 lemmer = nltk.stem.WordNetLemmatizer()
 def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
 remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
 def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


 from sklearn.feature_extraction.text import CountVectorizer
 LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
 LemVectorizer.fit_transform(dataAll)


print (LemVectorizer.vocabulary_)


tf_matrix = LemVectorizer.transform(documents).toarray()
print (tf_matrix)


 from sklearn.feature_extraction.text import TfidfTransformer
 tfidfTran = TfidfTransformer(norm="l2")
 tfidfTran.fit(tf_matrix)
 print (tfidfTran.idf_)
 
 tfidf_matrix = tfidfTran.transform(tf_matrix)
print (tfidf_matrix.toarray())

cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print (cos_similarity_matrix)


from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
def cos_similarity(textlist):
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()
cos_similarity(dataAll)




