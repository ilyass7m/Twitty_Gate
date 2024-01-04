from twitter_setup.twitterConnectionSetup import  twitter_setup
from twitter_setup.collect_tweets import get_tweets , collect_comments
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tweepy

api = twitter_setup()

import joblib
import numpy as np

# Load TfidfVectorizer, tfidf_matrix, and clf from the saved file
loaded_model_data = joblib.load(r'C:\Users\HOME\twitty_gate\SVM\model_data.joblib')

# Accessing loaded objects from the dictionary
loaded_tfidf_vectorizer = loaded_model_data['tfidf_vectorizer']
loaded_tfidf_matrix = loaded_model_data['tfidf_matrix']
loaded_clf = loaded_model_data['clf']

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']






def predict_svm(new_text):
    

    # Preprocessing steps similar to what was done during training
    punctuation = set(string.punctuation)
    doc = ''.join([w for w in new_text.lower() if w not in punctuation])
    doc = [w for w in doc.split() if w not in stopwords]
    stemmer = PorterStemmer()
    doc = [stemmer.stem(w) for w in doc]
    doc = ' '.join(w for w in doc)



    # Transform the new text to a TF-IDF representation
    new_text_tfidf = loaded_tfidf_vectorizer.transform([doc])

    # Use the trained classifier to predict the polarity of the new text
    predicted_polarity = loaded_clf.predict(new_text_tfidf)

    # Display the predicted polarity (1 or -1)
    return predicted_polarity[0]


def rate(username):

    positive_comments = 0
    total_comments = 0

    # Retrieve 100 tweets from the specified username
    tweets = tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode='extended', count=100).items()

    for tweet in tweets:
        if not tweet.retweeted:
            # Fetch 10 comments for each tweet
            comments = tweepy.Cursor(api.search, q='to:' + username, since_id=tweet.id, tweet_mode='extended').items(10)

            for comment in comments:
                # Use the 'predict_svm' function to predict the polarity of the comment
                polarity = predict_svm(comment.full_text)

                # Increment counters based on polarity prediction
                total_comments += 1
                if polarity == 1:
                    positive_comments += 1

    # Calculate the rate of positive comments
    if total_comments > 0:
        positive_rate = (positive_comments / total_comments) * 100
        return positive_rate
    else:
        return 0




