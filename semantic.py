import os
import pandas as pd
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
file_name = "data_set.csv"

df = pd.read_csv(file_name)
df = df.dropna()
df["tone"] = df["tone"].replace("Neative", "Negative")
df = df.drop_duplicates()
# Set target and test
x = df.drop(columns=["tone"]) #X
y = df["tone"] #y
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print(test)
categories = ["Positive", "Negative", "Neutral"]
target_train, target_test, test_train, test_test = train_test_split(x, y, stratify=y, random_state=0)
# print(test_train)
# train and test dataframe
train_data = pd.merge(target_train.reset_index(drop=True),
                    pd.DataFrame(test_train, columns=["tone"]),
                    left_index=True, right_index=True)

test_data = pd.merge(target_test.reset_index(drop=True),
                   pd.DataFrame(test_test, columns=["tone"]),
                   left_index=True, right_index=True)