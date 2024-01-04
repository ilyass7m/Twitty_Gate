import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc



############

with open('sentiment_analysis.csv', newline='', encoding='utf-8') as csv_file:
    # Create a CSV reader object
    reader = csv.DictReader(csv_file)
    data=[]
    labels=[]
    
    # Iterate through each row in the CSV file
    for row in reader:
        data.append(row['tweet'])

        

        if row['label'] == '0':

            labels.append(1)

        else: 
            labels.append(-1)



print(labels)



#%%
# data preprocessing
############
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

# For each document in the dataset, do the preprocessing
for doc_id, text in enumerate(data):
    
    # Remove punctuation and lowercase
    punctuation = set(string.punctuation)    
    doc = ''.join([w for w in text.lower() if w not in punctuation])
        
    # Stopword removal
    doc = [w for w in doc.split() if w not in stopwords]  
        
    # Stemming
    stemmer = PorterStemmer()
    doc = [stemmer.stem(w) for w in doc] 
        
    # Covenrt list of words to one string
    doc = ' '.join(w for w in doc)
    data[doc_id] = doc       


#%%
# create the TF-IDF matrix
#############
m = TfidfVectorizer()
tfidf_matrix = m.fit_transform(data)
tfidf_matrix = tfidf_matrix.toarray() # convert to numpy array
print ("Size of TF-IDF matrix: ", tfidf_matrix.shape )   
print ("Sparsity of the TF_IDF matrix: ", \
float(np.count_nonzero(tfidf_matrix)) / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))

data_train, data_test, labels_train, labels_test = train_test_split(tfidf_matrix, labels, test_size=0.4, random_state=42)




#clf = svm.SVC(C=1.5, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#clf = LogisticRegression()
clf = svm.LinearSVC()
#clf = BernoulliNB().fit(data_train,labels_train)
#clf = RandomForestClassifier(n_estimators=100)
y_score = clf.fit(data_train, labels_train)
labels_predicted = clf.predict(data_test)   
       
# Evaluation of the prediction
print (classification_report(labels_test, labels_predicted))
print ("The accuracy score is {:.2%}".format(accuracy_score(labels_test, labels_predicted)))

import joblib

# Assuming 'tfidf_vectorizer', 'tfidf_matrix', and 'clf' are your objects
# Save TfidfVectorizer, tfidf_matrix, and clf to a single file (e.g., 'model_data.joblib')
model_data = {
    'tfidf_vectorizer': m,
    'tfidf_matrix': tfidf_matrix,
    'clf': clf
}
joblib.dump(model_data, 'model_data.joblib')

