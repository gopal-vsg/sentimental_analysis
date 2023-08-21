import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
analiser  =  SentimentIntensityAnalyzer()

filename = input("enter the review file name : ")
with open(filename, 'r') as file:
    text = file.read()

sentiment_scores = sid.polarity_scores(text)
if sentiment_scores['compound'] >= 0.05:
    sentiment = "positive"
elif sentiment_scores['compound'] <= -0.05:
    sentiment = "negative"
else:
    sentiment = "neutral"

print("the review feels to be : ",sentiment)