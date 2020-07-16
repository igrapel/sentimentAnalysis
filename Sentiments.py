import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#load positive reviews into DF
dir = '/home/ilan/aclImdb/train/ttP'
os.chdir(dir)

data = []
files = [file for file in os.listdir(dir) if os.path.isfile(file)]
for file in files:
  with open (file, "r") as myfile:
    data.append(myfile.read())

df1 = pd.DataFrame(data)
df1['Sentiment']=1

#load negative reviews into DF
dir = '/home/ilan/aclImdb/train/ttN'
os.chdir(dir)
data = []
files = [file for file in os.listdir(dir) if os.path.isfile(file)]
for file in files:
  with open (file, "r") as myfile:
    data.append(myfile.read())

df2 = pd.DataFrame(data)
df2['Sentiment']=0

df = df1.append(df2)

#Tokenize words
tokened_reviews = [word_tokenize(rev) for rev in df[0]]

# Create feature that measures length of reviews
len_tokens = []

for i in range(len(tokened_reviews)):
     len_tokens.append(len(tokened_reviews[i]))

#Create word stems
stemmed_tokens = []
porter = PorterStemmer()
for i in range(len(tokened_reviews)):
  stems = [porter.stem(token) for token in tokened_reviews[i]]
  stems = ' '.join(stems)
  stemmed_tokens.append(stems)

# Add features of stems and the lengh of each review (need to convert stems to numerical values)
df.insert(1, column='Stemmed', value=stemmed_tokens)
df.insert(2, column='TokenLength', value=len_tokens)

#Create bag of words features
#Not yet in df
vect = CountVectorizer(max_features=100, ngram_range=(1,3), stop_words=ENGLISH_STOP_WORDS)
ReviewBOW = vect.fit_transform(df[0])
BOW_df=pd.DataFrame(ReviewBOW.toarray(), columns=vect.get_feature_names())
print("BOW:  ", BOW_df.shape)

#Create bag of words for stems features
#Not yet in df
vectStem = CountVectorizer(max_features=100, ngram_range=(1,3), stop_words=ENGLISH_STOP_WORDS)
StemBOW = vectStem.fit_transform(df['Stemmed'])
StemBOW_df=pd.DataFrame(StemBOW.toarray(), columns=vect.get_feature_names())
print("StemBOW:  ", StemBOW_df.head())

#Create TfIdf features
#Not yet in DF
Tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=100, stop_words=ENGLISH_STOP_WORDS)
TfidfReview = Tfidf.fit_transform(df[0])

Tfidf_df = pd.DataFrame(TfidfReview.toarray(), columns=vect.get_feature_names())
print("TFIDF")
print(Tfidf_df.tail())

df.columns = ['Review', 'TokenLength', 'Stemmed', 'Sentiment' ]
print("DF", df.shape)
print(df.columns)
