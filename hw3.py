#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
from sklearn.svm import SVC
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    lower_case_text = text.lower()
    text_without_possessive = re.sub(r"(\w+)'s", r'\1', lower_case_text)
    text_without_apostrophes = text_without_possessive.replace("'", "")
    text_without_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text_without_apostrophes)
    cleaned_text = re.sub(r'[^\w\s…]', ' ', text_without_urls)
    tokenized_words = nltk.word_tokenize(cleaned_text)

    lemmatized_words = []
    for word, tag in nltk.pos_tag(tokenized_words):
        wordnet_pos_tag = posMapping.get(tag[0].upper(), 'n')  
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos_tag)
        lemmatized_words.append(lemmatized_word)

    return lemmatized_words
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df['text'] = df['text'].apply(lambda x: process(x, lemmatizer))
    return df
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    stop_words_collection = list(stop_words)
    
    def tokenize_custom(tokens):
        return tokens
    
    vectorizer_tfidf = TfidfVectorizer(tokenizer=tokenize_custom,
                                       lowercase=False,
                                       stop_words=stop_words_collection,
                                       min_df=2)
    
    transformed_features = vectorizer_tfidf.fit_transform(processed_tweets['text'])

    return vectorizer_tfidf, transformed_features

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    def label_map(screen_name):
        if screen_name in ['realDonaldTrump', 'mike_pence', 'GOP']:
            return 0
        else:
            return 1

    labels = processed_tweets['screen_name'].apply(label_map).to_numpy()
    return labels
#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.majority_class = None

    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        self.majority_label = statistics.mode(y)

    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        y_prediction = [self.majority_label for _ in X]
        return y_prediction

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
    classifier = SVC(kernel=kernel)
    classifier.fit(X_train, y_train)
    return classifier

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    predictions = classifier.predict(X_validation)
    accuracy = accuracy_score(y_validation, predictions)
    return accuracy

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    unlabeled_tweets = process_all(unlabeled_tweets)
    features = tfidf.transform(unlabeled_tweets['text'])
    predicted_labels = classifier.predict(features)

    return predicted_labels