import pandas as pd
import numpy as np

data = pd.read_csv('data_moviereviews.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_texts = x_train[:, 0]
x_test_texts = x_test[:, 0]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train_texts)
x_test = vectorizer.transform(x_test_texts)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train[:, 0])

predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_test[:, 0], predictions)
report = classification_report(y_test[:, 0], predictions)

print("Accuracy: ",acc)
print("Classification Report:\n", report)

file_name = input("upload the moview review file : ")

with open(file_name, 'r') as file:
    content = file.read()
content_vectorized = vectorizer.transform([content])


ans = classifier.predict(content_vectorized)
if ans[0]==1:
    result = 'negitive'
else:
    result = 'positive'

print("Predicted Sentiment:", result)