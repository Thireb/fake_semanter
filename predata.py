# USUAL Imports
import os
import pandas as pd
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from icecream import ic
import re
from stopwords_urdu import STOPWORDS
import numpy as np
################ Data Fixing ##########################

file_name = "data_set.csv"

df = pd.read_csv(file_name)
df = df.dropna()
df["tone"] = df["tone"].replace("Neative", "Negative")
df = df.drop_duplicates()
ic(df.head)
ic(df.shape)

##################################################################################
######### Performing X-Y categorization #################

target = df.drop(columns=["tone"])  # X
test = df["tone"]  # y
encoder = LabelEncoder()
test = encoder.fit_transform(test)
categories = ["Positive", "Negative", "Neutral"]

# Convert 'target' and 'test' back to Pandas Series
target_series = pd.Series(target.values.squeeze())
test_series = pd.Series(test)

# Handle missing values (if any)
target_series.fillna(target_series.mode()[0], inplace=True)
test_series.fillna(test_series.mode()[0], inplace=True)

# Check the shapes of the input variables
ic(target_series.shape)
ic(test_series.shape)

# Ensure the number of samples is the same
if target_series.shape[0] != test_series.shape[0]:
    raise ValueError(
        "Number of samples in 'target_series' and 'test_series' is not the same."
    )

# Check for classes with only one member
unique_classes = pd.unique(test_series)
single_member_classes = [
    c for c in unique_classes if test_series.tolist().count(c) == 1
]

if single_member_classes:
    ic("Classes with only one member:", single_member_classes)
    # Remove the classes with only one member
    for c in single_member_classes:
        test_series.pop(test_series[test_series == c].index[0])



##################### training testing Splitting #############################



target_series.pop(target_series.index[0])
test_series.name = "tone"
target_series.name = "comments"

# Re-create the train-test split
target_train, target_test, test_train, test_test = train_test_split(
    target_series, test_series, stratify=test_series, random_state=0
)
ic("test_train", test_train)
ic("target_train", target_train)


# train and test dataframe
train_data = pd.merge(
    target_train.reset_index(drop=True),
    pd.DataFrame(test_train, columns=["tone"]),
    left_index=True,
    right_index=True,
)

test_data = pd.merge(
    target_test.reset_index(drop=True),
    pd.DataFrame(test_test, columns=["tone"]),
    left_index=True,
    right_index=True,
)
# Including Pre processing in the Code block Below
# corpus of words
# remove stopwords
def create_meta_features(df):
    """This function creates meta features for EDA analysis"""

    # word count
    df["word_count"] = df["comments"].apply(lambda x: len(str(x).split()))
    # unique_word_count
    df["unique_word_count"] = df["comments"].apply(lambda x: len(set(str(x).split())))
    # stop_word_count
    df["stop_word_count"] = df["comments"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS])
    )
    # mean_word_length
    df["mean_word_length"] = df["comments"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()])
    )
    # char_count
    df["char_count"] = df["comments"].apply(lambda x: len(str(x)))
    # Some text are just the space character, which gives NaN values for mean_word_length
    df = df.fillna(0)  # Fill NaNs

    return df

main_df = create_meta_features(df)


def make_corpus(target):
    corpus = []
    ic(len(target_train))
    for i in range(0, len(target_train)):
        review = re.sub("[^a-zA-Z]", " ", target_train.iloc[i])
        review = review.lower()
        review = review.split()
        review = [word for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)
    return corpus

corpus = make_corpus(target_train)

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
target_train_tfidf = tfidf_vectorizer.fit_transform(corpus)
# ic(dir(target_test))
# ic(target_test.to_list)
target_test_tfidf = tfidf_vectorizer.transform(target_test)
ic(target_train_tfidf, target_test_tfidf)


############## TFIDF CONVERSION ##############



