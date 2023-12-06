#!/usr/bin/env python
# coding: utf-8

# Description::
# 
# The main objective of the project is to detect malware in files that are downloaded from online. The dataset was acquired from kaggle.com where the tables are labeled based on their features and a class label differentiating malware and benign files.
# 
# The project is using deep learning technique such as (ANN) for the analysis and later will be compared with three machine learning models on their accuracy and precision. Confusion matrix will be used as well for measurement on how each model performend.
# The acquired dataset will be divided into three parts. One for training the model, another for testing the model and the final part is to validate the model.

# In[ ]:





# In[ ]:





# Step 1 :
# Importing librraries like Pandas, Numpy, Seaborn, sklearn, Tensor-flow, Keras.

# In[1]:


from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
from sklearn.metrics import pair_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict 
import seaborn as sns;sns.set(font_scale=1.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
import seaborn as sns
from tensorflow.keras import layers
from tensorflow import keras
import keras_tuner


# In[ ]:





# Step 2:
# LOAD DATASET

# In[2]:


Data = pd.read_csv(r"C:/Users/hp/OneDrive/Desktop/Projectdataset/Malware dataset.csv")


# Step 3
# UNDERSTAND YOUR DATA BEFORE BEGIN ANY ANALYSIS?
# Data Exploration Analysis
# 
# Exploratory Data Analysis(EDA) which is commonly known as Data Exploration. 
# It is one of the major Data Analysis Processes that deals with various techinques
# to explore the data presented to understand the dataset.
# 
# For this project, i perform the various steps listed below to explore the dataset
# to understand the type of data i'm using for my analysis.
# 
# 

# 2.1) Understanding the dataset
# 
#     2.1. Head and Tail of the dataset
#     2.2. The shape of the dataset
#     2.3. Data colunms present
#     2.4. Info of the Dataset
#     2.5. Summary of the Dataset 

# # 2.1. The Shape of the Dataset
# The shape function display the total shape of the dataset consisting of rows
# and colunms
# 

# In[3]:


#Displaying the total rows and columns
Data.shape


# ROWS = 100000
# COLUNMS = 35

# # 2.2. Head and Tail of the Dataset Listing
# 
# The head and Tail functions from the dataset display the top 5 and the last 5 
# data from the dataset records.
# This function head(i),tail(i) shows the first i rows of data based on their positional arrangement. they are much important in data analysis to check for
# the right data type.
# 

# In[4]:


#Displays the first five Data records
Data.head()


# In[5]:


#Displays the last five Data records
Data.tail()


# In[6]:


#The head function provides an array that can allow a manual inserting values
#into the head specifying the numbers you want to see from the dataset.
Data.head(7)


# # 2.3. List the Data types within the dataset
# The dtype function list all the available colunms with their data type in the dataset

# In[7]:


#Listing the data types of all the colunms within the dataset
Data.dtypes


# # 2.4 Information of the Dataset
# The info() function is one of the various functions that is used to check for the information about the data as well as the datatypes and their respective attributes

# In[8]:


# Finding the total information about the dataset and its colunms and also find 
#out if there is any null values count.
Data.info()


# # 2.5. The Dataset Summary Using Statistical Means
# The dataset summary uses a statistical means such as describe to spread the numerical values within the dataset. Its displays the min,max,mean and percentile values of the data
# 

# In[9]:


#Using Statistical Summary of the dataset
Data.describe()


# In[10]:


#Displaying the content of the mean values for all the colunms
Data.mean()


# In[11]:


print(Data['usage_counter'].mean())


# In[12]:


#displaying the minimum values of the dataset
Data.min()


# In[13]:


Data.std()


# In[ ]:





# # 3.0. Data Pre-processing
# 
# Data pre-processing is the major concepts for processing the acquired dataset as part of the data pre-processing methods design for this project work. 
# Preparing raw data for machine learning models is known as data preparation. A machine-learning model can be created starting with this step. This component of data science requires the most time and complexity. Machine learning algorithms must preprocess data in order to simplify them.
# Numerous issues can arise with data in the actual world. It might overlook some components or details. Data preprocessing's main goal is to correct and improve the data to make it more valuable, even while incomplete or absent data is utterly useless. 
# 

# # 3.1. Data Cleaning
# The data cleaning processes were to check and look for null values, duplicates, and missing values in the given dataset and remove such values as such for further pre-processing.
# 
#     3.1.1. Detecting Duplicates and Remove such
#     3.1.2. Detecting Null values and remove such or use the following to fill the null values
#     the following will be used to fill the null values
#     #data = data.fillna(0)
#     #data = data.fillna(mean_of_column)
#     3.1.3. Droping unneccesary attributes
#     3.1.4. Data formating from categorical to numeric form
#     for this work, the i will only perform data formating on the 'classification' which is the class label for represent the   
#     outcome of the detection whether malware or benign.
#     

# In[14]:


#Detecting duplicates values in the dataset
Data.duplicated().sum(axis=0)


# There were 0 duplicates values for the given dataset

# In[15]:


Data.isnull().sum(axis=0).sum()


# In[16]:


Data.isna().sum().sum(axis=0)


# There is no null value or missing values registed to this particular dataset

# In[17]:


#graphical presentation of missing values
import missingno as msno
msno.matrix(Data.iloc[:,: 35])


# # 3.1.4 Dropping Irrelevant features/attributes
# Dropping features that are irrelevant to the model is established based on two main reason decided by the proposed model and they are;
# 
#        1. The feature should have an influence in the dataset but not just descriptions such as hashing etc.
#        2. The feature selected should not have no zero values through-out the entire rows.
# Figure 4.6 shows the removal results of all the 11 irrelevant features detected by the data cleaning processes.
# 

# In[18]:


data = Data.drop(['signal_nvcsw','normal_prio','policy','vm_pgoff','cgtime','nr_ptes','hiwater_rss','cached_hole_size','task_size','usage_counter','hash'],axis=1)


# 4.3.2. Dropping Irrelevant Data Features 
# Dropping features that are irrelevant to the model
# 

# In[19]:


data.shape


# # 3.1.4.B. Detecting / Replacing Zero Values 
# 

# 3.1.2. Checking the number of Zeros OF 7 FEATURE

# In[20]:


print('The number of zeros in fs_excl_counter =', data[data['fs_excl_counter']==0].shape[0])
print('The number of zeros in millisecond =', data[data['millisecond']==0].shape[0])
print('The number of zeros in state =', data[data['state']==0].shape[0])
print('The number of zeros in nivcsw =', data[data['nivcsw']==0].shape[0])
print('The number of zeros in gtime =', data[data['gtime']==0].shape[0])
print('The number of zeros in min_flt =', data[data['min_flt']==0].shape[0])
print('The number of zeros in free_area_cache =', data[data['free_area_cache']==0].shape[0])


# In[21]:


#Replacing the zero values with the mean value for each column and printing out the results afterwards 


# In[22]:


data['millisecond'] = data['millisecond'].replace(0,data['millisecond'].mean())
print('The number of zeros in millisecond =', data[data['millisecond']==0].shape[0])
data['fs_excl_counter'] = data['fs_excl_counter'].replace(0,data['fs_excl_counter'].mean())
print('The number of zeros in fs_excl_counter =', data[data['fs_excl_counter']==0].shape[0])
data['state'] = data['state'].replace(0,data['state'].mean())
print('The number of zeros in state =', data[data['state']==0].shape[0])
data['nivcsw'] = data['nivcsw'].replace(0,data['nivcsw'].mean())
print('The number of zeros in nivcsw =', data[data['nivcsw']==0].shape[0])
data['gtime'] = data['gtime'].replace(0,data['gtime'].mean())
print('The number of zeros in gtime =', data[data['gtime']==0].shape[0])
data['min_flt'] = data['min_flt'].replace(0,data['min_flt'].mean())
print('The number of zeros in min_flt =', data[data['min_flt']==0].shape[0])
data['free_area_cache'] = data['free_area_cache'].replace(0,data['free_area_cache'].mean())
print('The number of zeros in free_area_cache =', data[data['free_area_cache']==0].shape[0])



# # 3.1.4 Data Formating
# Coverting the categorical letters to numeric values for easier tabulations by DL/ML models.
# For the purposes of DL/ML models, such data-types needs to be encoded into numeric format since such models work with mathematical equations. Therefore, the class label which represent the classification was encoded unto numeric values 

# In[23]:


#Counting the values from the class label of the dataset
data['classification'].value_counts()


# In[24]:


d= {k:i for i, k in enumerate(data['classification'].unique(),0)}
data['classification'] = data['classification'].map(d)


# In[25]:


#checking the status after converting the class label
data['classification'].value_counts()


#     Now 1 represent Benign
#     and 0 represent Malware

# In[26]:


data.head()


# In[27]:


#Graphical presentation of the class label
f,ax=plt.subplots(1,2,figsize=(10,5))
data['classification'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],shadow=True)
ax[0].set_title('Class Label')
ax[0].set_ylabel('')
sns.countplot(data,x='classification',ax=ax[1])
ax[1].set_title('Class Label')
M,B = data['classification'].value_counts()
print('Malware  (0): ',M)
print('Benign   (1): ',B)
plt.grid()
plt.show()


# # 4.0 Data Visualization
# Data visualization using the pie chart and the subplot

# In[ ]:





# # 4.1. Feature engineering
# Feature engineering is a series of techniques that is use to help improve a model learning processes by giving the learning algorithm an additional background experience in selecting relevant features for an optimal detection. The feature engineering techniques used by the thesis work is feature scaling, train-test and validation dataset splits and correlation analysis.
# The proposed model uses correlation function and heat-map to select variables that are highly correlated with the target label.
# 
#     

# # 4.1.1Correlation Analysis (Feature Selection)
# Correlation analysis was apply to these project to define the relationship between two variables that are closely related.
# 

# In[28]:


corrt = data.corr()
top_corr_ft = corrt.index
plt.figure(figsize=(25,10))
#plot heat map
k = sns.heatmap(data[top_corr_ft].corr(),annot=True,cmap='RdYlGn')


# From the correlation heat-map, the feature variables relating to the class label which is the classification registered 12 variables that have high correlation to the classification output out of the 24 total variables presented for this analysis. Therefore, the proposed model focus on using the 12 high correlated variables for it predictive model as an input data for the design of the model.
#             these 12 variables are             
#                         (utime,fs_excl_counter,maj_fit,nivcsw,nvcsw,end_data,reserved_vm,exec_vm,shared_vm,map_count,vm_trucate_count,static_prio)

# In[29]:


data = data.drop(['gtime','stime','lock','min_flt','last_interval','total_vm','mm_users','free_area_cache','prio','state','millisecond'],axis=1)


# In[30]:


data.shape


# In[ ]:





# # 4.1.2 Dataset Spliting into train,test,validation,set

# In[31]:


##splitting the dataset into train and test set to build out the x and y axis.
## split the dataset into 70% for training and 30% for testing
train_dataset, test_dataset = train_test_split(data,test_size=0.3)


# In[32]:


train_dataset.shape,test_dataset.shape


# In[33]:


## Split the test dataset into 50% test set and 50% validation set
## Building out the x-axis
test_data,validation_data = train_test_split(test_dataset,test_size=0.5)


# In[34]:


test_data.shape, validation_data.shape


# In[35]:


trainset=train_dataset.describe(include='all')
trainset.pop('classification')
trainset=trainset.transpose()
trainset


# In[36]:


## Building out the labels from each dataset
## Building out the y axis

train_labels = train_dataset.pop('classification')
test_labels = test_data.pop('classification')
validation_labels = validation_data.pop('classification')


# In[37]:


print(train_labels.value_counts(),'as Train_labels'),
print(test_labels.value_counts(),'as Test_labels'),
print(validation_labels.value_counts(),'as Validation_labels')



# where 0 = represent Malware(M)
# where 1 = represent Benign(B)

# # 4.1.3. Data Normalization
# ## Statistics on the training dataset

# In[38]:


#DATA NORMALIZATION 
# First form
## Define a function to normalize the dataset
def norm(x):
    return (x - trainset['mean']) / trainset['std']
# Return the function on the actual dataset
norm_train_dataset = norm(train_dataset)
norm_test_dataset = norm(test_data)
norm_validation = norm(validation_data)



# In[39]:


norm_train_dataset.head()


# In[40]:


norm_test_dataset.head()


# In[ ]:





# In[ ]:





# data=Data.describe(include='all')
# data.pop('classification')
# data = data.transpose()
# data

# In[ ]:





# In[ ]:





# data = data.fillna(0)

# data.head(5)

# In[41]:


epochs = 10


# In[42]:


from tensorflow.keras import optimizers


# In[43]:


from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout


# In[44]:


def build_model_five_hidden_layers():
    model = Sequential()
    
    # theis will help build the keras model having 32 neurons and 5 hidden layers
    model.add(Dense(32,input_shape=(norm_train_dataset.shape[1],)))
    #model.add(Activation('relu'))
    
    model.add(Dense(32, Activation('relu')))
    
    model.add(Dense(32, Activation('relu')))
    
    model.add(Dense(32,Activation('relu')))
    
    model.add(Dense(16,Activation('relu')))
    
    model.add(Dense(16,Activation('relu')))
    
    #model.add(Dense(1,Activation('LeakyReLU')))
    model.add(Dense(1))
    
    ## Activation: sigmod, softmax, tanh, relu, LeakyReLU
    #Optimizer: SGD, Adam, RMSProp
    learning_rate = 0.1
    optimizer = optimizers.SGD(learning_rate)
    
    #model.compile(optimizer=optimizer,loss='mae',metrics=['mae','mse','mape'])
    
    model.compile(optimizer=optimizer,loss='mae',metrics=['accuracy'])
    
    return model


# In[45]:


model = build_model_five_hidden_layers()
model.summary()


# In[46]:


history=model.fit(norm_train_dataset,train_labels,batch_size=32,epochs=epochs,validation_data=(norm_test_dataset,test_labels))


# In[47]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail(5)


# In[48]:


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
train_val_acc = history.history['val_accuracy']
xc = range(epochs)
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(['train_acc','Train_val_acc'])
plt.style.use(['classic'])


# In[49]:


print('Train Split :')
loss,accuracy = model.evaluate(norm_train_dataset,train_labels,verbose=3)

print('Accuracy  : {:5.2f}'.format(accuracy))


# In[50]:


print('Test Split :')
loss,accuracy = model.evaluate(norm_test_dataset,test_labels,verbose=3)

print('Accuracy  : {:.5f}'.format(accuracy))


# In[51]:


print('Validation Split :')
loss,accuracy = model.evaluate(norm_validation,validation_labels,verbose=3)

print('Accuracy  : {:.4f}'.format(accuracy))


# In[52]:


# MODEL EVALUATION
eva_pred = model.predict(norm_validation)
eva_pred = [1 if y>= 0.5 else 0 for y in eva_pred]

accuracy_score(validation_labels,eva_pred)


# In[53]:


# MODEL TESTING
test_pred = model.predict(norm_test_dataset)
test_pred = [1 if y>= 0.5 else 0 for y in test_pred]

accuracy_score(test_labels,test_pred)


# In[54]:


from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,GradientBoostingRegressor


# models={
#     "Logistic Regression":LogisticRegression(max_iter=1000,),
#     "Naive Bayes": GaussianNB(var_smoothing=4,priors=None),
#     "Kneighbor": KNeighborsClassifier(),
#     "Support Vector":SVC(),
#    #"Random Forest":RandomForestClassifier(criterion='entropy',random_state=42,max_depth=40,n_jobs=-1),
#     #'Decision tree':DecisionTreeClassifier(criterion='entropy',splitter='best',random_state=42,max_depth=40),
#     'xgb':GradientBoostingClassifier(n_estimators=25,max_depth=50,min_samples_split=500,min_samples_leaf=40,max_features='sqrt',subsample=0.8,learning_rate=0.5),
#     #'xgbregress':GradientBoostingRegressor(n_estimators=20,max_depth=4,random_state=42)
#     #'adb':AdaBoostClassifier()
# }
#     
# for i in range(len(list(models))):
#     model = list(models.values())[i]
#     model.fit(norm_train_dataset, train_labels)  #train model
#     
#     # make predictions
#     y_train_pred = model.predict(norm_train_dataset)
#    
#     
#     # Training set performance
#     model_train_classification =classification_report(train_labels,y_train_pred)
#     model_train_accuracy = accuracy_score(train_labels, y_train_pred)
#     model_train_f1 = f1_score(train_labels, y_train_pred, average='weighted')
#     model_train_precision = precision_score(train_labels, y_train_pred)
#     model_train_recall = recall_score(train_labels, y_train_pred)
#     
#     print(list(models.keys())[i])
#     
#     print('Model Performance for Training set')
#     print(model_train_classification)
#     print("- Accuracy: {:.4f}".format(model_train_accuracy))
#     print("- F1 score: {:.4f}".format(model_train_f1))
#     
#     print("- Precision: {:.4f}".format(model_train_precision))
#     print("- Recall: {:.4f}".format(model_train_recall))
#    
#     
#     print('.........................................')
#     
#     
#     print('='*35)
#     print('\n')

# In[ ]:





# # BUIDING MACHINE LEARNING CLASSIFIERS

# In[55]:


## Model creation using support vector classifier

svc =SVC()
#train model
svc.fit(norm_train_dataset,train_labels)
# make predictions using the training dataset.
#svc_predict_train = svc.predict(norm_train_dataset)

#svc_train_classification = classification_report(train_labels,svc_predict_train)
#svc_train_accuracy = accuracy_score(train_labels, svc_predict_train)
#svc_train_f1 = f1_score(train_labels, svc_predict_train)
#svc_train_precision = precision_score(train_labels, svc_predict_train)
#svc_train_recall = recall_score(train_labels, svc_predict_train)
#svc_train_rocauc_score = roc_auc_score(train_labels, svc_predict_train)
    
#print('Model Performance for Training set')
#print(svc_train_classification)
#print("- Accuracy: {:.4f}".format(svc_train_accuracy))
#print("- F1 score: {:.4f}".format(svc_train_f1))
    
#print("- Precision: {:.4f}".format(svc_train_precision))
#print("- Recall: {:.4f}".format(svc_train_recall))
#print("- Roc Auc score: {:.4f}".format(svc_train_rocauc_score))
    
#print('.........................................')

    
    
    
    
#print('='*35)
#print('\n')


# In[56]:


## Model Creation
KN =KNeighborsClassifier()
#train model
KN.fit(norm_train_dataset,train_labels)
# make predictions
#KN_predict_train = KN.predict(norm_train_dataset)

#KN_train_classification = classification_report(train_labels,KN_predict_train)
#KN_train_accuracy = accuracy_score(train_labels, KN_predict_train)
#KN_train_f1 = f1_score(train_labels, KN_predict_train)
#KN_train_precision = precision_score(train_labels, KN_predict_train)
#KN_train_recall = recall_score(train_labels, KN_predict_train)
#KN_train_rocauc_score = roc_auc_score(train_labels, KN_predict_train)

#print('Model Performance for Training set on KNN')
#print(KN_train_classification)
#print("- Accuracy: {:.4f}".format(KN_train_accuracy))
#print("- F1 score: {:.4f}".format(KN_train_f1))
    
#print("- Precision: {:.4f}".format(KN_train_precision))
#print("- Recall: {:.4f}".format(KN_train_recall))
#print("- Roc Auc score: {:.4f}".format(KN_train_rocauc_score))

#print('.........................................')

    
    
    
    
#print('='*40)
#print('\n')


# In[57]:


## Model Creation
LR =LogisticRegression()
#train model
LR.fit(norm_train_dataset,train_labels)
# make predictions
#LR_predict_train = KN.predict(norm_train_dataset)

#LR_train_classification = classification_report(train_labels,LR_predict_train)
#LR_train_accuracy = accuracy_score(train_labels, LR_predict_train)
#LR_train_f1 = f1_score(train_labels, LR_predict_train)
#LR_train_precision = precision_score(train_labels, LR_predict_train)
#LR_train_recall = recall_score(train_labels, LR_predict_train)
#LR_train_rocauc_score = roc_auc_score(train_labels, LR_predict_train)

#print('Model Performance for Training set on Linear Regression')
#print(KN_train_classification)
#print("- Accuracy: {:.4f}".format(LR_train_accuracy))
#print("- F1 score: {:.4f}".format(LR_train_f1))
    
#print("- Precision: {:.4f}".format(LR_train_precision))
#print("- Recall: {:.4f}".format(LR_train_recall))
#print("- Roc Auc score: {:.4f}".format(LR_train_rocauc_score))

#print('.........................................')

    
    
    
    
#print('='*40)
#print('\n')


# # MAKING PREDICTION BASED ON THE TEST DATASET¶
# 

# In[58]:


LR_predict_test = LR.predict(norm_test_dataset)

acc_LR = accuracy_score(LR_predict_test,test_labels)

print("- Accuracy Linear Regression: {:.4f}".format(acc_LR))


# In[59]:


KN_predict_test = KN.predict(norm_test_dataset)
acc_KN =accuracy_score(KN_predict_test,test_labels)
print("- Accuracy KNeighbour: {:.3f}".format(acc_KN))


# In[60]:


svc_predict_test = svc.predict(norm_test_dataset)
acc_SVC = accuracy_score(svc_predict_test,test_labels)
print("- Accuracy Support Vector: {:.3f}".format(acc_SVC))


# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn import metrics


# In[62]:


KNN_fpr, KNN_tpr, threshold = metrics.roc_curve(test_labels,KN_predict_test)
Sv_fpr, Sv_tpr, threshold = metrics.roc_curve(test_labels,svc_predict_test)
LR_fpr, LR_tpr, threshold = metrics.roc_curve(test_labels,LR_predict_test)
plt.figure(figsize=(6,10),dpi=130)

roc_auc_Sv = metrics.auc(Sv_fpr,Sv_tpr)


disp = RocCurveDisplay.from_predictions(y_true=test_labels,y_pred=svc_predict_test,name='SVC')
RocCurveDisplay.from_predictions(test_labels,KN_predict_test,name='KNN',ax=disp.ax_)
RocCurveDisplay.from_predictions(test_labels,LR_predict_test,name='LR',ax=disp.ax_)
plt.plot([0,1],[0,1],color='darkblue',linestyle='--')


plt.legend(loc='lower right')
plt.title('Model Evaluation Using ROC_AUC CURVE')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[63]:


KN =KNeighborsClassifier()
#train model

# make predictions
#KN_predict_test = KN.predict(norn_test_dataset)

KN_cm = confusion_matrix(test_labels,KN_predict_test)
cm_KN = (np.round(KN_cm/np.sum(KN_cm,axis=1).reshape(-1,1),2))



KN_test_accuracy = accuracy_score(test_labels,KN_predict_test)


print("- Accuracy: {:.4f}".format(KN_test_accuracy))
print(KN_cm)

sns.heatmap(cm_KN,cmap='plasma',annot=True,xticklabels=[0,1],yticklabels=[0,1])
plt.title('KNearest Neighbor')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[ ]:





# In[64]:


SV = SVC()
#train model



SVC_cm = confusion_matrix(test_labels,svc_predict_test)
cm_SV = (np.round(SVC_cm/np.sum(SVC_cm,axis=1).reshape(-1,1),2))



SVC_test_accuracy = accuracy_score(test_labels,svc_predict_test)


print("- Accuracy: {:.4f}".format(SVC_test_accuracy))
print(KN_cm)

sns.heatmap(cm_SV,cmap='flag',annot=True,xticklabels=[0,1],yticklabels=[0,1])
plt.title('Support Vector')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[65]:


LR_cm = confusion_matrix(test_labels,LR_predict_test)
cm_LR = (np.round(LR_cm/np.sum(LR_cm,axis=1).reshape(-1,1),2))
LR_test_accuracy = accuracy_score(test_labels,LR_predict_test)
print("- Accuracy: {:.4f}".format(LR_test_accuracy))
print(KN_cm)
sns.heatmap(cm_LR,cmap='terrain',annot=True,xticklabels=[0,1],yticklabels=[0,1])
plt.title('Learn Regression')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[ ]:





# # VALIDATING THE MODEL BASED ON VALIDATION DATASET

# In[66]:


LR_predict_val = LR.predict(norm_validation)

acc_LR = accuracy_score(LR_predict_val,validation_labels)

print("- Accuracy Linear Regression: {:.4f}".format(acc_LR))


# In[70]:


KN = KNeighborsClassifier()

KN.fit(norm_validation,validation_labels)

KN_predict_val = KN.predict(norm_validation)
acc_KN =accuracy_score(KN_predict_val,validation_labels)
print("- Accuracy KNeighbour: {:.3f}".format(acc_KN))


# In[68]:


svc_predict_val = svc.predict(norm_validation)
acc_SVC = accuracy_score(svc_predict_val,validation_labels)
print("- Accuracy Support Vector: {:.3f}".format(acc_SVC))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




