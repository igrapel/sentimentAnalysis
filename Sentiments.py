import os
import pandas as pd
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Nice to see additional metrics
from sklearn.metrics import classification_report

# Defining your constants up top to change easily
path_training = '/home/ilan/aclImdb/train'
path_testing = '/home/ilan/aclImdb/test'

def load_data(dir):
    # Open and import positve data
    posDir = os.chdir(dir + '/pos')
    df1 = files_to_DF(posDir)
    df1['Sentiment'] = 1  # Setting whether it is a pos or neg review
    # Open and import negative data
    negDir = os.chdir(dir + '/neg')
    df2 = files_to_DF(negDir)
    df2['Sentiment'] = 0  # Setting whether it is a pos or neg review
    #join positive and negative into a dataframe
    df = df1.append(df2)

    df.columns = ['Review', 'Sentiment']  # Labeling columns
    # Remove non-alphanumeric characters
    df['Review'] = df['Review'].apply(lambda x: re.sub("[^a-zA-Z]", ' ', str(x)))
    # Tokenize the training and testing data
    df_tokenized = tokenize_review(df)
    return df_tokenized

def files_to_DF(dir):
    data = []
    files = [file for file in os.listdir(dir) if os.path.isfile(file)]
    for file in files:
        with open(file, "r") as myfile:
            data.append(myfile.read())
    df = pd.DataFrame(data)
    return df

def tokenize_review(df):
    # Tokenize Reviews in training
    tokened_reviews = [word_tokenize(rev) for rev in df['Review']]
    # Create word stems
    stemmed_tokens = []
    porter = PorterStemmer()
    for i in range(len(tokened_reviews)):
        stems = [porter.stem(token) for token in tokened_reviews[i]]
        stems = ' '.join(stems)
        stemmed_tokens.append(stems)
    df.insert(1, column='Stemmed', value=stemmed_tokens)
    return df

def transform_BOW(training, testing, stemmed_or_unstemmed):
    vect = CountVectorizer(max_features=10000, ngram_range=(1,3), stop_words=ENGLISH_STOP_WORDS)
    vectFit = vect.fit(training[stemmed_or_unstemmed])
    BOW_training = vectFit.transform(training[stemmed_or_unstemmed])
    BOW_training_df = pd.DataFrame(BOW_training.toarray(), columns=vect.get_feature_names())
    BOW_testing = vectFit.transform(testing[stemmed_or_unstemmed])
    BOW_testing_Df = pd.DataFrame(BOW_testing.toarray(), columns=vect.get_feature_names())
    return BOW_training_df, BOW_testing_Df

def transform_tfidf(training, testing, stemmed_or_unstemmed):
    Tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=10000, stop_words=ENGLISH_STOP_WORDS)
    Tfidf_fit = Tfidf.fit(training[stemmed_or_unstemmed])
    Tfidf_training = Tfidf_fit.transform(training[stemmed_or_unstemmed])
    Tfidf_training_df = pd.DataFrame(Tfidf_training.toarray(), columns=Tfidf.get_feature_names())
    Tfidf_testing = Tfidf_fit.transform(testing[stemmed_or_unstemmed])
    Tfidf_testing_df = pd.DataFrame(Tfidf_testing.toarray(), columns=Tfidf.get_feature_names())
    return Tfidf_training_df, Tfidf_testing_df

def add_augmenting_features(df):
    tokened_reviews = [word_tokenize(rev) for rev in df['Review']]
    # Create feature that measures length of reviews
    len_tokens = []
    for i in range(len(tokened_reviews)):
        len_tokens.append(len(tokened_reviews[i]))
    len_tokens = preprocessing.scale(len_tokens)
    df.insert(0, column='Lengths', value=len_tokens)

    # Create average word length (training)
    Average_Words = [len(x)/(len(x.split())) for x in df['Review'].tolist()]
    Average_Words = preprocessing.scale(Average_Words)
    df['averageWords'] = Average_Words
    return df

def build_model(X_train, y_train, X_test, y_test, name_of_test):
    log_reg = LogisticRegression(C=30).fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print('Training accuracy of '+name_of_test+': ', log_reg.score(X_train, y_train))
    print('Testing accuracy of '+name_of_test+': ', log_reg.score(X_test, y_test))
    print(classification_report(y_test, y_pred))  # Evaluating prediction ability

# Load training and test sets
# Loading reviews into DF
df_train = load_data(path_training)

print('...successfully loaded training data')
print('Total length of training data: ', len(df_train))
# Add augmenting features
df_train = add_augmenting_features(df_train)
print('...augmented data with len_tokens and average_words')
# print(df_train.head())

# Load test DF
df_test = load_data(path_testing)

print('...successfully loaded testing data')
print('Total length of testing data: ', len(df_test))
# Add augmenting features
df_test = add_augmenting_features(df_test)
print('...augmented data with len_tokens and average_words')
# print(df_test.head())

# Create unstemmed BOW features for training set
df_train_bow_unstem, df_test_bow_unstem = transform_BOW(df_train, df_test, 'Review')
print('...successfully created the unstemmed BOW data')

# Create stemmed BOW features for training set
df_train_bow_stem, df_test_bow_stem = transform_BOW(df_train, df_test, 'Stemmed')
print('...successfully created the stemmed BOW data')

print('...successfully created the stemmed BOW data')

# Create TfIdf features for training set
df_train_tfidf_unstem, df_test_tfidf_unstem = transform_tfidf(df_train, df_test, 'Review')
print('...successfully created the unstemmed TFIDF data')

# Create TfIdf features for training set
df_train_tfidf_stem, df_test_tfidf_stem = transform_tfidf(df_train, df_test, 'Stemmed')

print('...successfully created the stemmed TFIDF data')

# Running logistic regression on dataframes
build_model(df_train_bow_unstem, df_train['Sentiment'], df_test_bow_unstem, df_test['Sentiment'], 'BOW Unstemmed')
build_model(df_train_bow_stem, df_train['Sentiment'], df_test_bow_stem, df_test['Sentiment'], 'BOW Stemmed')

build_model(df_train_tfidf_unstem, df_train['Sentiment'], df_test_tfidf_unstem, df_test['Sentiment'], 'TFIDF Unstemmed')
build_model(df_train_tfidf_stem, df_train['Sentiment'], df_test_tfidf_stem, df_test['Sentiment'], 'TFIDF Stemmed')
