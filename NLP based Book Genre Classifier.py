#!/usr/bin/env python
# coding: utf-8

# Initial Import Libraries


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# @author Arpan Sarkar
# PGP c5dbab9480173ef71df02e9b7721aa65277588f102243d14b1ed1ed0c38a09ef

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
print("Libraries Loaded...")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import Data from Dataset

# In[ ]:


data = pd.read_csv('books.csv', encoding='ISO-8859-1')


# Assign and Select Columns

# In[ ]:


columns = ['id','image','link', 'book_name', 'author','no','genre']
data.columns = columns


# Use Columns required and Perform Data Cleaning

# In[ ]:


books = pd.DataFrame(data['book_name'])
genre = pd.DataFrame(data['genre'])
#Cleaning
data['book_name'] = data['book_name'].fillna('No Book')

print("Data Loaded")
# In[ ]:


#print (len(books))
#print (len(genre))
#verifying

#Print Unique Genres
# In[ ]:


genre['genre'].unique()


# Using Label Encoder to binarize data

# In[ ]:


from sklearn.preprocessing import LabelEncoder

#feat = ['genre']
#for x in feat:
   # le = LabelEncoder()
   # le.fit(list(genre[x].values))
    #genre[x] = le.transform(list(genre[x]))
#binarized labels
#genre['genre'].unique()

print("Encoding Complete..")
# Using Global Verifier
import joblib
# In[ ]:
le = joblib.load('label.pkl')

#le.inverse_transform([0])[0]


# In[ ]:


data['everything'] = pd.DataFrame(data['book_name'])
#print (data['everything'].head(5))


# Importing NLP stopwords

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
#stop[:10]


# Removing Special Charecters, Expressions and Stopwords

# In[ ]:


def change(t):
    t = t.split()
    return ' '.join([(i) for (i) in t if i not in stop])

data['everything'].apply(change)


# Using ID Vectorizer from Tensor Nodes for text feature extraction

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')
vectors = vectorizer.fit_transform(data['everything'])
vectors.shape

print("Vectorizing Complete")
# Generating Train and Test Sets

# In[ ]:


#from sklearn import metrics
#from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split 

#X_train, X_test, y_train, y_test = train_test_split(vectors, genre['genre'], test_size=0.15)

#print (X_train.shape)
#print (y_train.shape)
#print (X_test.shape)
#print (y_test.shape)


# Using MLP back propagation network for classification

# In[ ]:


from sklearn.neural_network import MLPClassifier

import joblib
#testing
clf = joblib.load('books.pkl')
print("Model Loaded Successfully")
#clf = MLPClassifier(activation='relu', alpha=0.00003, batch_size='auto',
                   #beta_1=0.9, beta_2=0.999, early_stopping=True,
                   #epsilon=1e-08, hidden_layer_sizes=(20,), learning_rate='adaptive',
                   #learning_rate_init=0.004, max_iter=200, momentum=0.9,
                   #nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                   #solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=True,
                   #warm_start=False)

#clf.fit(X_train, y_train)

#pred = clf.predict(X_test)
#print (metrics.f1_score(y_test, pred, average='macro'))
#print (metrics.accuracy_score(y_test, pred))


# Checking Linearirty of results

# In[ ]:


#plt.ylabel('cost')
#plt.xlabel('iterations')
#plt.plot(clf.loss_curve_)
#plt.plot(pose_clf.loss_curve_)
#plt.show()


# Testing the Model

# In[ ]:


text = input("Enter the book name \n")
text = [text,]
s = (vectorizer.transform(text))
#s = vectorizer.fit_transform(df)
print (s.shape)
d = (clf.predict(s))
print(le.inverse_transform(d)[0])


# Saving the model

# In[ ]:


