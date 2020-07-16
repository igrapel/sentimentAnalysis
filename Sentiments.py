import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Load training and test sets
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
#Join positive and negative into same dataframe
df = df1.append(df2)

#load positive reviews into test DF
dir = '/home/ilan/aclImdb/test/testPos'
os.chdir(dir)

data_test = []
files = [file for file in os.listdir(dir) if os.path.isfile(file)]
for file in files:
  with open (file, "r") as myfile:
    data_test.append(myfile.read())

df1 = pd.DataFrame(data_test)
df1['Sentiment']=1

#load negative reviews into test DF
dir = '/home/ilan/aclImdb/test/testNeg'
os.chdir(dir)
data_test = []
files = [file for file in os.listdir(dir) if os.path.isfile(file)]
for file in files:
  with open (file, "r") as myfile:
    data_test.append(myfile.read())

df2 = pd.DataFrame(data_test)
df2['Sentiment']=0
#Join positive and negative into one test dataframe
df_test = df1.append(df2)

#Tokenize Reviews in training
tokened_reviews = [word_tokenize(rev) for rev in df[0]]
#Create word stems
stemmed_tokens = []
porter = PorterStemmer()
for i in range(len(tokened_reviews)):
  stems = [porter.stem(token) for token in tokened_reviews[i]]
  stems = ' '.join(stems)
  stemmed_tokens.append(stems)
df.insert(1, column='Stemmed', value=stemmed_tokens)
#Tokenize Review in Test
tokened_reviews_test = [word_tokenize(rev) for rev in df_test[0]]
#Create word stems
stemmed_tokens_test = []
porter = PorterStemmer()
for i in range(len(tokened_reviews_test)):
  stems = [porter.stem(token) for token in tokened_reviews_test[i]]
  stems = ' '.join(stems)
  stemmed_tokens_test.append(stems)
df_test.insert(1, column='Stemmed', value=stemmed_tokens_test)

df.columns = ['Review', 'Stemmed', 'Sentiment']
df_test.columns = ['Review', 'Stemmed', 'Sentiment']

#Create unstemmed BOW features for training set
vect = CountVectorizer(max_features=100, ngram_range=(1,3), stop_words=ENGLISH_STOP_WORDS)
vectFit = vect.fit(df['Review'])
BOW = vectFit.transform(df['Review'])
BOW_df=pd.DataFrame(BOW.toarray(), columns=vect.get_feature_names())

#Create unstemmed BOW features for test set
BOW_test = vectFit.transform(df_test['Review'])
BOW_test_df=pd.DataFrame(BOW_test.toarray(), columns=vect.get_feature_names())
#print("BOW:  ", BOW_df.shape)

#Create stemmed BOW features for training set
vect = CountVectorizer(max_features=100, ngram_range=(1,3), stop_words=ENGLISH_STOP_WORDS)
vectFit = vect.fit(df['Stemmed'])
BOW_stemmed = vectFit.transform(df['Stemmed'])
BOW_stemmed_df=pd.DataFrame(BOW_stemmed.toarray(), columns=vect.get_feature_names())

#Create stemmed BOW features for test set
BOW_stemmed_test = vectFit.transform(df_test['Stemmed'])
BOW_stemmed_test_df=pd.DataFrame(BOW_stemmed_test.toarray(), columns=vect.get_feature_names())
#print("BOW:  ", BOW_df.shape)

#Create TfIdf features
Tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=100, stop_words=ENGLISH_STOP_WORDS)
Tfidf_fit = Tfidf.fit(df['Review'])
Tfidf_trans = Tfidf_fit.transform(df['Review'])
Tfidf_df = pd.DataFrame(Tfidf_trans.toarray(), columns=Tfidf.get_feature_names())
#For Testset
Tfidf_trans_test = Tfidf_fit.transform(df_test['Review'])
Tfidf_df_test = pd.DataFrame(Tfidf_trans_test.toarray(), columns=Tfidf.get_feature_names())
labels_training = df['Sentiment']
labels_testing = df_test['Sentiment']

#Create TfIdf stemmed features
Tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=100, stop_words=ENGLISH_STOP_WORDS)
Tfidf_stemmed_fit = Tfidf.fit(df['Stemmed'])
Tfidf_stemmed_trans = Tfidf_stemmed_fit.transform(df['Stemmed'])
Tfidf_stemmed_df = pd.DataFrame(Tfidf_stemmed_trans.toarray(), columns=Tfidf.get_feature_names())
#For Testset
Tfidf_stemmed_trans_test = Tfidf_stemmed_fit.transform(df_test['Stemmed'])
Tfidf_stemmed_df_test = pd.DataFrame(Tfidf_stemmed_trans_test.toarray(), columns=Tfidf.get_feature_names())

labels_training = df['Sentiment']
labels_testing = df_test['Sentiment']



print(df.columns)
# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(BOW_df, labels_training)
print('Accuracy of logistic regression (BOW features): ', log_reg.score(BOW_df, labels_training))
print('Accuracy of logistic regression test (BOW features): ', log_reg.score(BOW_test_df, labels_testing))

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(BOW_stemmed_df, labels_training)
print('Accuracy of logistic regression (stemmed BOW features): ', log_reg.score(BOW_stemmed_df, labels_training))
print('Accuracy of logistic regression test (stemmed BOW features): ', log_reg.score(BOW_stemmed_test_df, labels_testing))

# Build a logistic regression model and calculate the accuracy on Tfidf
log_reg = LogisticRegression().fit(Tfidf_df, labels_training)
print('Accuracy of logistic regression (Tfidf features): ', log_reg.score(Tfidf_df, labels_training))
print('Accuracy of logistic regression test (Tfidf features): ', log_reg.score(Tfidf_df_test, labels_testing))

# Build a logistic regression model and calculate the accuracy on Tfidf stemmed
log_reg = LogisticRegression().fit(Tfidf_stemmed_df, labels_training)
print('Accuracy of logistic regression (Tfidf Stemmed features): ', log_reg.score(Tfidf_stemmed_df , labels_training))
print('Accuracy of logistic regression test (Tfidf Stemmed features): ', log_reg.score(Tfidf_stemmed_df_test, labels_testing))
