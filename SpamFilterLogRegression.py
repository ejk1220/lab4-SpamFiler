# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:51:44 2019

@author: jdk450
"""

import os
import emailReadUtility
import pandas  as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


os.getcwd()
DATA_DIR = "C:\\Users\\Evan\\Downloads\\trec07p\\trec07p\\data"
LABELS_FILE = "C:\\Users\\Evan\\Downloads\\trec07p\\trec07p\\full\\index"
TESTING_SET_RATIO = 0.2

labels = {}
# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0
        
def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = emailReadUtility.extract_email_text(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
        
    return X, y

X, y = read_email_files()

#take a look at X and y . Look at the individual emails and index file to make sense of what you see.
pd.DataFrame(X).head()
pd.DataFrame(y).head()

X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)), 
    train_size=TESTING_SET_RATIO, random_state=2)

vectorizer = TfidfVectorizer()
X_train_vector= vectorizer.fit_transform(X_train)
X_test_vector= vectorizer.transform(X_test)


# Initialize the classifier and make label predictions
cl_lr = LogisticRegression()
cl_lr.fit(X_train_vector, y_train)
y_pred = cl_lr.predict(X_test_vector)

#get confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

#show confusion matrix
print(cnf_matrix)

# compute and Print performance metrics
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print(" Logistic Regression Precision:", precision_score(y_test, y_pred))
print("Logistic Regression Recall:", recall_score(y_test, y_pred))
print("Logistic Regression F1 score:",f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Create Decision tree classifer 
treeClassifier = DecisionTreeClassifier()
#Train Classifer 
treeClassifier = treeClassifier.fit(X_train_vector, y_train)
y_treePred = treeClassifier.predict(X_test_vector)
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_treePred)))
print("Decision Tree Precision:", precision_score(y_test, y_treePred))
print("Decision Tree Recall:", recall_score(y_test, y_treePred))
print("Decision Tree F1 Score:", f1_score(y_test, y_treePred))
print(classification_report(y_test, y_treePred))
treecnf_matrix = confusion_matrix(y_test, y_treePred)
print(treecnf_matrix)

#Create Support Vector Classifier 
sv_clf = svm.SVC(kernel='linear')
#Train Classifier 
sv_clf = sv_clf.fit(X_train_vector, y_train)
y_svpred = sv_clf.predict(X_test_vector)
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_svpred)))
print("Support Vector Precision:", precision_score(y_test, y_svpred))
print("Support Vector Recall:", recall_score(y_test, y_svpred))
print("Support Vector F1 Score:", f1_score(y_test, y_svpred))
print(classification_report(y_test, y_svpred))
svcnf_matrix = confusion_matrix(y_test, y_svpred)
print(svcnf_matrix)

#Create Random Forest Classifier 
forest_clf = RandomForestClassifier()
#Train Classifier 
forest_clf = forest_clf.fit(X_train_vector, y_train)
y_forestpred = forest_clf.predict(X_test_vector)
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_forestpred)))
print("Random Forest Precision:", precision_score(y_test, y_forestpred))
print("Random Forest Recall:", recall_score(y_test, y_forestpred))
print("Random Forest F1 Score:", f1_score(y_test, y_forestpred))
print(classification_report(y_test, y_forestpred))
forestcnf_matrix = confusion_matrix(y_test, y_forestpred)
print(forestcnf_matrix)

#Create MLP Classifier 
mlp_clf = MLPClassifier(random_state=101)
mlp_clf = mlp_clf.fit(X_train_vector, y_train)
y_mlppred = mlp_clf.predict(X_test_vector)
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_mlppred)))
print("MLP Precision:", precision_score(y_test, y_mlppred))
print("MLP Recall:", recall_score(y_test, y_mlppred))
print("MLP F1 Score:", f1_score(y_test, y_mlppred))
print(classification_report(y_test, y_mlppred))
mlpcnf_matrix = confusion_matrix(y_test, y_mlppred)
print(mlpcnf_matrix)



