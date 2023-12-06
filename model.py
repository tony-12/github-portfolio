#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask
import pandas as pd
import bs4
import itertools
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.parse import urlparse
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
from sklearn.metrics import pair_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict 
import seaborn as sns;sns.set(font_scale=1.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, precision_score, recall_score,f1_score, roc_auc_score,roc_curve,auc
import missingno as msno
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn import metrics
from colorama import Fore  #Colorama is a module to color the python outputs
from pprint import pprint
from tld import get_tld, is_tld
from urllib.parse import urlparse


# In[2]:


data =pd.read_csv(r'urldata1.csv')


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.head(4)


# In[ ]:





# In[6]:


data.isnull().sum()


# In[7]:


data.duplicated().sum()


# In[8]:


df = data.drop_duplicates()


# In[9]:


df.shape


# In[10]:


df['Label'].value_counts()


# In[11]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[ ]:





# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)


# In[13]:


X_train.shape, X_test.shape,y_train.shape,y_test.shape

X_sd = StandardScaler()
X_train = X_sd.fit_transform(X_train)
X_test = X_sd.fit_transform(X_test)
# In[14]:


dt =DecisionTreeClassifier()
#train model
dt.fit(X_train,y_train)
#model predicting based on the test dataset
dt_predict_test = dt.predict(X_test)

dt_test_classification = classification_report(y_test,dt_predict_test)
dt_test_accuracy = accuracy_score(y_test, dt_predict_test)
dt_test_f1 = f1_score(y_test, dt_predict_test)
dt_test_precision = precision_score(y_test, dt_predict_test)
dt_test_recall = recall_score(y_test, dt_predict_test)
dt_test_rocauc_score = roc_auc_score(y_test, dt_predict_test)

print('Model Performance for Testing the model')

print("- Accuracy: {:.4f}".format(dt_test_accuracy))
print("- F1 score: {:.4f}".format(dt_test_f1))
    
print("- Precision: {:.4f}".format(dt_test_precision))
print("- Recall: {:.4f}".format(dt_test_recall))
print("- Roc Auc score: {:.4f}".format(dt_test_rocauc_score))

print('.........................................')

    
    
    
    
print('='*40)
print('\n')


# In[15]:


dt_predict_test


# In[16]:


# 7. saving the model in pickle


# In[17]:


import pickle


# In[18]:


with open('DT_Model1.pickle','wb') as f:
    pickle.dump(dt,f)

with open('Scaler.pickle','wb') as f:
    pickle.dump(X_sd,f)
# In[ ]:




