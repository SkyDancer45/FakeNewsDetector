
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Reading data
data_fake = pd.read_csv("./Fake.csv")
data_true = pd.read_csv("./True.csv")

print("FAKE DATA ")
print(data_fake.head())
print("TRUE DATA")
print(data_true.head())

# Adding class labels
data_fake["class"] = 0
data_true["class"] = 1

# Inspect data
print(data_fake.shape, data_true.shape)

# Save last 10 entries for manual testing
data_fake_mannual_testing = data_fake.tail(10).copy()
data_true_mannual_testing = data_true.tail(10).copy()

# Remove last 10 entries from the main dataset
data_fake = data_fake.iloc[:-10, :]
data_true = data_true.iloc[:-10, :]

# Merge data
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(["title", "subject", "date"], axis=1)

# Text preprocessing function
def wordpot(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

# Apply text preprocessing
data["text"] = data["text"].apply(wordpot)

# Split data
x = data["text"]
y = data["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Text vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression model
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Classification Report")
print(classification_report(y_test, pred_lr))

# Decision Tree model
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Classification Report")
print(classification_report(y_test, pred_dt))

# Function for output label
def output_label(n):
    return 'Fake News' if n == 0 else 'Not Fake News'

# Manual testing function
def mannual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordpot)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    print(f"\n\nLR prediction: {output_label(pred_LR[0])} \nDT prediction: {output_label(pred_DT[0])}")

# Manual testing input
news = str(input("Enter a news text: "))
mannual_testing(news)

