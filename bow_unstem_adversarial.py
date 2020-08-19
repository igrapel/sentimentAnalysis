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
def load_data(df):
  df['Review'] = df['Review'].apply(lambda x: re.sub("[^a-zA-Z]", ' ', str(x)))
  # Tokenize the training and testing data
  df_tokenized = tokenize_review(df)
  return df_tokenized

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
    return vectFit, BOW_training_df, BOW_testing_Df

def transform_tfidf(training, testing, stemmed_or_unstemmed):
    Tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=10000, stop_words=ENGLISH_STOP_WORDS)
    Tfidf_fit = Tfidf.fit(training[stemmed_or_unstemmed])
    Tfidf_training = Tfidf_fit.transform(training[stemmed_or_unstemmed])
    Tfidf_training_df = pd.DataFrame(Tfidf_training.toarray(), columns=Tfidf.get_feature_names())
    Tfidf_testing = Tfidf_fit.transform(testing[stemmed_or_unstemmed])
    Tfidf_testing_df = pd.DataFrame(Tfidf_testing.toarray(), columns=Tfidf.get_feature_names())
    return Tfidf_fit, Tfidf_training_df, Tfidf_testing_df

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
    log_reg = LogisticRegression(C=30, max_iter=500).fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print('Training accuracy of '+name_of_test+': ', log_reg.score(X_train, y_train))
    print('Testing accuracy of '+name_of_test+': ', log_reg.score(X_test, y_test))
    print(classification_report(y_test, y_pred))  # Evaluating prediction ability
    return log_reg

df_train = pd.read_csv('https://raw.githubusercontent.com/igrapel/Python/master/RET/Data_Augmented.csv')
df_train = load_data(df_train)
df_train = add_augmenting_features(df_train)

# Load test
df_test = pd.read_csv('https://raw.githubusercontent.com/igrapel/Python/master/RET/bow_unstemmed_adversarial_examples.csv')
df_test = df_test.drop(['num_queries', 'original_output',
       'original_score', 'original_text', 'perturbed_output', 'perturbed_score', 'result_type'], axis=1)
df_test.columns = ['Sentiment', 'Review']
df_test = load_data(df_test)
df_test = add_augmenting_features(df_test)

# Create unstemmed BOW features for training set
unstemmed_BOW_vect_fit, df_train_bow_unstem, df_test_bow_unstem = transform_BOW(df_train, df_test, 'Review')
print('...successfully created the unstemmed BOW data')
# Running logistic regression on dataframes
bow_unstemmed = build_model(df_train_bow_unstem, df_train['Sentiment'], df_test_bow_unstem, df_test['Sentiment'], 'BOW Unstemmed')
