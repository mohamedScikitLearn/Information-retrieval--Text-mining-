                
#================================
# @ authors : BERRIMI Mohamed 
#  TP : Information retieval                      #
#   ----  VERSION ---1---       #
#===============================#   
from   nltk.stem     import PorterStemmer
from   nltk.tokenize import sent_tokenize, word_tokenize
from   nltk.corpus   import stopwords
from   nltk.tokenize import RegexpTokenizer
from   collections   import Counter
import nltk
import glob
import errno
import os 
import re # Regular Expressions 
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd  
import PyPDF2 # From PDF to text library 
import easytextract # From PDF, PNG to text library 
import numpy as np 
  
#===================================
#  Function Counting : This fucntion
# count the number of words in the 
#file and show them in Desc order.
# @param : Path of the corpus   
#                               
#==================================

def Counting(path):
    corpus = glob.glob(path)
    for name in corpus:
        
            with open(name) as f:

            #create a list of all words fetched from the file using a list comprehension
                words = [word for line in f for word in line.split()]
            print ("The total word count in file",name," is:", len(words))
            
            c = Counter(words)
            for word, count in c.most_common():
               print (word, count)
               
Counting('/Users/macbookair/Desktop/corpus/*.txt')
 
#=====================================================
#Function : word_Search : This function search for a word ,typed by the user ,
#in the corpus it test the existence of the word 
#in each file and each line of the file , and total count #
#@param : path of the corpus .  #
#===============================#

def Word_Search(path):
    # update the path here , from root to your folder that contains  the text files 
    path = '/Users/macbookair/Desktop/corpus/*.txt'
    
    corpus = glob.glob(path)
    string1  = input("Please enter the  word you are looking for  : ")
    for name in corpus:
        # for each file in the corpus 
        try:
            with open(name) as f: # open the file 
                print('You are looking in File  :', name)
                count = 0
                nblines=0
                for line in f:
                    nblines +=1 
                    if string1 in line:
                        count+=1
                        print('----found----------')
                    else:
                        print('-----The word doesnt exist in this line------- ')
                print("The word ",string1,"exist in this file :",count ,"times")
                print("Number of lines of this file   **********",nblines)   
                
        except IOError as exc: # if files doesn't exist , throw exeception 
            if exc.errno != errno.EISDIR:
                print('Files not found ')
# call the function             
Word_Search('/Users/macbookair/Desktop/corpus/*.txt') 
#========================================
#Function: StopWord_elem : this function count the number of words in the text file 
#of the corpus and return the count .
#The it applies stop word elimination on them and re calculate the count . 
#
# 
# @param : corpus path  
#                             
#========================================

def StopWordElem(path):
 
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    path = '/Users/macbookair/Desktop/corpus/*.txt'
    
    corpus = glob.glob(path)
    
    for name in corpus:

        data=open(name).read().replace('\n', '') #Sentence tokenize 
        words = word_tokenize(data) # word tokenizing 

        stopWords = set(stopwords.words('english')) 
        wordsFiltered = []
        for w in words:
                if w not in stopWords:
                    wordsFiltered.append(w)
         
        print ("The total word count in file :",name ,'is : ', len(words))
        print("Number of words after stop words elimination : ",len(wordsFiltered))
        print(wordsFiltered)
        
        print('========The most Frequent Words ======')
        words = [word for line in wordsFiltered for word in line.split()]
        print ("The total word count is:", len(words))
                
        c = Counter(words)
        for word, count in c.most_common():
            print (word, count)
            
StopWordElem('/Users/macbookair/Desktop/corpus/*.txt')

######################
#           OCR 
#  retrieve text from PDF file 
#  Note : update the path to the text file to test . 
#
###################


pdfFileObj = open('exemple.pdf','rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
num_pages = pdfReader.numPages
count = 0
text = ""
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

keywords = [word for word in nltk.word_tokenize(text)]
print(keywords)
for line in keywords :
    print(line)
    
    
    import re
#============   Version 2 ==============
import PyPDF2

pdfFileObj = open('exemple.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

print(pdfReader.numPages)

pageObj = pdfReader.getPage(0)

print(pageObj.extractText())

pdfFileObj.close()
""" Extract Text from PNG file"""######################
#           
#  retrieve text from PDF file 
#  Note : update the path to the text file to test . 
#
###################

from PIL import Image
from pytesseract import image_to_string
print (image_to_string(Image.open('image_with_text.png')))
print (image_to_string(Image.open('test-image_with_text.png'), lang='eng'))



"""" REMOVE THE DIGITS FROM WORDS """
import re
sentece='Hello mohame ded '
def containsDigit(sentece):
    if re.search("\d", sentece):
        return sentece
print(containsDigit(sentece))




#===================================
#  Function vocFile : This function constructs the vocabulary of each file in the corpus 
# and store the vocabulary of each file in a specific vecotr in the data array . 
# 
#
# @param : Path of the corpus   #
#                               #
#===============================#
    """ Vocabulary of each file in corpus  """
    def vocFile:
        # Update the path here 
        path = '/Users/macbookair/Desktop/corpus/*.txt'
        corpus = glob.glob(path)
        
        data=[]  # This array will store the vocabulary of each file 
        j=0
        y=0
        
        for name in corpus:
        
            data.append(open(name).read().replace('\n', ''))
           # for each file in the corpus , open it and append with data[]
           
           # Now the data[] contains the text files ( as arrays )
        
        for file in data:
        # For each file (vector ) in data []
        
            new_data=re.sub(r'\w*\d\w*', '', file) 
            # The re.sub == removes the digits and the ponctuations from the file . 
           
            tokenizer = RegexpTokenizer(r'\w+') 
        #Word Tokenizing 
            file=tokenizer.tokenize(new_data)
            
            # Stop word elimination 
            stopWords = set(stopwords.words('english'))
            # some stopwords haven't been eliminated , we manually added them 
            other = ['And', 'I','of','In','in','a','was',
                     'two','and','the','your','her','his','has',
                     'to','he ','that','As',
                     'in','The','Not','He',
                     'We','But','one','tooo']
      # Remove all the stop-words from the files              
            for word in list(file): 
                if word in other:
                    file.remove(word)
                    
            for word in list(file):
                if word in stopWords:
                    file.remove(word)
  
          data[j]=file # ==> each case in data[] contain a vector of each file 
            j=j+1   #move to the next file 
     # Thanks to Mohamed Rahmani 
lmtzr = WordNetLemmatizer()

for name in data:
    #for each file in data[]
    # Lematization
    data[y]=[lmtzr.lemmatize(word) for word in name]
    y=y+1


#===================================
#  Function AllVocabulary  : This function constructs the vocabulary of All the  files in the corpus 
# and store the vocabulary in AllVocstemed []
#
# @param :  #
# This function has 2 versions: 
#                               #
#===============================#

############
# VERSION 1 
#The main idea is :
# 1 Concatinate the vocabularies of all the previous file in one array 
# 2 : Remove redundancies 
###########


Complete = [] # Initilize an empty array 
for name in data:
    for word in name:
        Complete.append(word)
    len(Complete)
from collections import OrderedDict

All =[]
All  = list(OrderedDict.fromkeys(Complete))
len(All
############
# VERSION 2 
#
# The idea is : Apply the same thins as VocFile function , but in this time , stores the vocab in ine array .
###########


def AllVoab(path):
    corpus = glob.glob(path)
    
    dataAll=[]
    for name in corpus:
       
        dataAll.append(open(name).read().replace('\n', ''))
       
    for i in dataAll:
        
        new_data=re.sub(r'\w*\d\w*', '', i)
        #remove digits
        words = word_tokenize(new_data) 
        tokenizer = RegexpTokenizer(r'\w+')
        words=tokenizer.tokenize(new_data)


        stopWords = set(stopwords.words('english'))
       
        
        for word in list(words):  # iterating on a copy since removing will mess things up
            if word in other:
                words.remove(word)
                
                wordsFiltered = []
        
            for w in words:
                    if w not in stopWords:
                         
                        wordsFiltered.append(w)
                        
                        
                        AllVocStemed=[]
lmtzr = WordNetLemmatizer()
AllVocStemed = [lmtzr.lemmatize(word) for word in All]
AllVocStemed  = list(OrderedDict.fromkeys(AllVocStemed))


### Store the vocabulary in a text file . 
with open('vocabulaire.txt', 'w') as f:
    for word in AllVocStemed:
        f.write("%s\n" % word)

################ VSM


#================================
# Construct the Boolean matrix  #
#                               #
#                               #
#===============================#




### Boolean Matrix   
 #  The size of matrix == number of files in the corpus 
Matric_vocab=[[]]*len(data)

k=0
for k in range(len(data)):
    #initialize the matrix of the file X with a AllVocStemed as a header and values with 0
    Matric_vocab[k] = dict.fromkeys(AllVocStemed,0)

# look if the word in Vocabulary exists in file 
# if Yes , put the value of that word =1 in the  matrix 
for word in AllVocStemed:
    i=0
    for name in data:
        if word in name:
            Matric_vocab[i][word]=1
        i=i+1 
        
# using pandas to create  the dataFrame 
import pandas as pd
DataFrBinaire =pd.DataFrame(Matric_vocab, index=['file1','file2','file3','file4','file5','file6'])
#================================
# Construct the Indidence  matrix #
#                               #
#                               #
#===============================#

Matric_vocabMultp=[[]]*len(data)
k=0
for k in range(len(data)):
    # initilize an empty matrix with AllVocStemmed vocabulary as its header 
    Matric_vocabMultp[k] = dict.fromkeys(AllVocStemed,0)


for word in AllVocStemed:
    z=0
    for name in data: # for each file in corpus (vocabulary of file )
        for w in name: # for each word in vocabulary file  
            if w==word :
                # if the word exsits , increment its count by 1 . 
                Matric_vocabMultp[z][word]+=1
        z=z+1    
        
import pandas as pd
# use pandas to create the data frame 
DataFrameMult =pd.DataFrame(Matric_vocabMultp)


#  Store the dataframe into a csv file . 
DataFrameMult.to_csv('Matrix.csv',index=False , encoding='utf-8')


########## TF IDF ########
#================================
#  Function : TF : this function computes the Term-frequency 
# of the word in the document 
#                                 #
#                               #
#===============================#    

def computeTF (wordDict , file):
    
    tfDict ={}
    fileCount  = len(file ) 
    for word , count in wordDict.items(): 
        tfDict[word] = count/float(fileCount)
    return tfDict

# exemple : 
tf1 = computeTF(Matric_vocabMultp[0],data[0])
#================================
#  Function : computeIDF : this function computes the IDF 
# of the word in the document 
#                                 #
# @param : Matric_vocabMultp                              #
#===============================#   




def computeIDF(Matric_vocabMultp):
    import math 
    idfDict={}
    N=len(AllVocStemed)
    
    idfDict = dict.fromkeys(Matric_vocabMultp[0],0)
    for doc in Matric_vocabMultp:
        for word, val in doc.items():
            if val > 0 : 
                idfDict[word] +=1 
    for word , val in idfDict.items():
        if val > 0:
            idfDict[word]=math.log10(N/float(val))
    
    return idfDict

idfs = computeIDF([Matric_vocabMultp[1],Matric_vocabMultp[0],Matric_vocabMultp[3],Matric_vocabMultp[2],Matric_vocabMultp[4],Matric_vocabMultp[5]])




idfs = computeIDF([Matric_vocabMultp[1],Matric_vocabMultp[0],Matric_vocabMultp[3],Matric_vocabMultp[2],Matric_vocabMultp[4],Matric_vocabMultp[5]])




#================================
#  Compute TFIDF 
#@param : TF , IDF               
#                               #
#===============================#

def computeTFIDF (TF , idfs ):
    tfidfs={}
    for word  , val in TF.items():
        tfidfs[word]= val * idfs[word]
    return tfidfs 


TFIDF1 = computeTFIDF(tf1,idfs)
TFIDF2 = computeTFIDF(tf2,idfs)
TFIDF3 = computeTFIDF(tf3,idfs)
TFIDF4 = computeTFIDF(tf4,idfs)
TFIDF5 = computeTFIDF(tf5,idfs)
TFIDF6 = computeTFIDF(tf6,idfs)

import pandas as pd 

FrameTFIDF = pd.DataFrame ([TFIDF1,TFIDF2,TFIDF3,TFIDF4,TFIDF5,TFIDF6])



#================================
#  Normalizing The query 
#                               #
#                               #
#===============================#


Clean_query =[]
# Receive the user input . 
query  = input("Please enter your Query  : ")
# Tokenize the query  
tokenizer = RegexpTokenizer(r'\w+')
query=tokenizer.tokenize(query)
# Stop words elimination . 
stopWords = set(stopwords.words('english'))
 

for word in query:
    if word in  stopWords:
        query.remove(word)
# Lematization phasse 
for word in query : 
    q = lmtzr.lemmatize(word) 
    Clean_query.append(q)

#================================
#  Creat a vector initilized with 
#vocabulary words               #
#                               #
#===============================#


    # Initilize a vectore with Vocabulary as its header 
        Quer= dict.fromkeys(AllVocStemed,0)
        
        # Check if the words of the vocabulary exists in the query . 
        
        for word in AllVocStemed:
            for w in Clean_query:
              # If The word exists , increment the occurence in the vector  by 1 .   
                if word == w:
                    Quer[w]+=1
                
#================================
#  concatinate the query(vector )  with the Indicence Matrix 
#                                 #
#                               #
#===============================#    

# Now we are going to concatinate the query(vector )  with the Indicence Matrix 

Matric_vocaQuery=[[]]*len(data)

k=0
for k in range(len(data)):
    Matric_vocaQuery[k] = dict.fromkeys(AllVocStemed,0)


for word in AllVocStemed:
    z=0
    for name in data:
        for w in name:
            if w==word :
                # if the word exsits , increment its count by 1 . 
                Matric_vocaQuery[z][word]+=1
        z=z+1    

Matric_vocaQuery.append(Quer)
    
DataFrameQuery =pd.DataFrame(Matric_vocaQuery, index=['file1','file2','file3','file4','file5','file6','Query'])

########## TF IDF Of the QUERY ########
# We have already implimented  the functions , and the Matric_vocab_Query 
#  Calculate the TF of the query 
tfQ = computeTF(Matric_vocaQuery[6],Clean_query)

idfs = computeIDF([Matric_vocabMultp[1],
                   Matric_vocabMultp[0],
                   Matric_vocabMultp[3],
                   Matric_vocabMultp[2],
                   Matric_vocabMultp[4],
                   Matric_vocabMultp[5],
                   Matric_vocaQuery[6]])

TFIDF7 = computeTFIDF(tf7Q,idfs)
FrameTFIDFQuery= pd.DataFrame ([TFIDF1,TFIDF2,TFIDF3,TFIDF4,TFIDF5,TFIDF6,TFIDF7], index=['file1','file2','file3','file4','file5','file6','Query'])

### Cosin similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


vec = TfidfVectorizer()
X = vec.fit_transform(dataAll) # `X` will now be a TF-IDF representation of the data, the first row of `X` corresponds to the first sentence in `data`

# Calculate the pairwise cosine similarities (depending on the amount of data that you are going to have this could take a while)
CS = cosine_similarity(X)

