#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Importing Packages'''
import pandas as pd
import numpy as nm
import pandas_profiling as pf


# In[2]:


'''Importing Data'''
data = pd.read_csv('./flightdata.csv')
data.head()


# In[3]:


'''EDA'''
data.isnull().sum()


# In[4]:


data.describe()


# In[5]:


data = data.drop('Unnamed: 25', axis = 1)


# In[6]:


data.isnull().sum()


# In[7]:


'''Selecting Data Necessary for creating Model'''
data = data[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15", "DISTANCE"]]


# In[8]:


data[data.isnull().values.any(axis = 1)]


# In[9]:


'''Target Variable definition'''

data = data.fillna({'ARR_DEL15':1})
data.iloc[177:185]


# In[10]:


'''Featuring engineering'''

import math

for index, row in data.iterrows():
    data.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
data.head()


# In[11]:


'''Creating Dummies of features'''
train_data = pd.get_dummies(data , columns = ['ORIGIN', 'DEST'])


# In[12]:


train_data.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x = train_data.drop(['ARR_DEL15'], axis =1)
y = train_data['ARR_DEL15']


# In[15]:


x.shape, y.shape


# In[16]:


'''Splitting Dataset into train and test data'''

x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size = .33, random_state = 42 )


# In[17]:


'''Importing Machine Learning Packages'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial


# In[25]:


'''Applying Different Machine Learning Classification Models'''

def models(X,Y,x,y):

#     Logistinc Regression
    log = LogisticRegression(random_state=0)
    log.fit(X,Y)
    log.predict(x)
    
#     Decision Tree
    des_tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
    des_tree.fit(X,Y)
    des_tree.predict(x)
    
#     Knearestneighbour
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p =2)
    knn.fit(X,Y)
    knn.predict(x)
    
#     Random Forrest Model
    forrest = RandomForestClassifier(random_state=0)
    forrest.fit(X,Y)
    forrest.predict(x)
    
# Using SVC method of svm class to use Support Vector Machine Algorithm
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X, Y)
    svc_lin.predict(x)

#Using SVC method of svm class to use Kernel SVM Algorithm
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X, Y)
    svc_rbf.predict(x)

#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
    gauss = GaussianNB()
    gauss.fit(X, Y)
    gauss.predict(x)
    
#     XGBoost Classifier
    xg = XGBClassifier()
    xg.fit(X,Y)
    xg.predict(x)
    
    print('Logistic Regression Training Accuracy {}, Validation Accuracy {}:'.format(log.score(X, Y), log.score(x,y)))
    print('K Nearest Neighbor Training Accuracy {}: Validation Accuracy {}'.format(knn.score(X, Y), knn.score(x,y)))
    print('Support Vector Machine (Linear Classifier) Training Accuracy {}: Validation Accuracy {}'.format(svc_lin.score(X, Y), svc_lin.score(x,y)))
    print('Support Vector Machine (RBF Classifier) Training Accuracy {}: Validation Accuracy {}'.format(svc_rbf.score(X, Y),svc_rbf.score(x,y) ))
    print('Gaussian Naive Bayes Training Accuracy {}: Validation Accuracy {}'.format(gauss.score(X, Y), gauss.score(x,y)))
    print('Decision Tree Classifier Training Accuracy {}: Validation Accuracy {}'.format(des_tree.score(X, Y),des_tree.score(x,y) ))
    print('Random Forest Classifier Training Accuracy {}: Validation Accuracy {}'.format(forrest.score(X, Y), forrest.score(x,y)))  
    print('XGBoost Classifier Training Accuracy {}: Validation Accuracy {}'.format(xg.score(X, Y), xg.score(x,y)))
    return log, des_tree, forrest, svc_lin, svc_rbf, gauss, xg


# In[26]:


model(x_train, y_train, x_test, y_test)


# In[32]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[33]:


# Instantiate the classfiers and make a list
classifiers = [KNeighborsClassifier(n_neighbors=5, metric='minkowski', p =2), 
               DecisionTreeClassifier(criterion='entropy',random_state=0), 
               RandomForestClassifier(random_state=0), 
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(x_train, y_train)
    yproba = model.predict_proba(x_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


# In[37]:


import matplotlib.pyplot as plt
import numpy as np


# In[38]:


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# In[48]:


'''Selecting XGBoost Classifier which has higher AUC - Better model'''


# In[47]:


model = XGBClassifier()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("Training Accuracy is {} and Validation Accuracy is {}".format(model.score(x_train, y_train),
                                                                     model.score(x_test, y_test)))


# In[40]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[44]:


print(confusion_matrix(y_test, predicted))
print(precision_score(y_test, predicted))
print(recall_score(y_test, predicted))


# In[45]:


'''Python function'''


# In[46]:


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()
    
    if (origin == 'ATL' or origin=='SEA') and (destination == 'SEA' or destination == 'ATL'):
        distance = 2182.0
    elif (origin == 'DTW' or origin=='MSP') and (destination == 'MSP' or destination == 'STW'):
        distance = 528.0
    elif (origin == 'SEA' or origin=='MSP') and (destination == 'MSP' or destination == 'SEA'):
        distance = 1399.0
    elif (origin == 'SEA' or origin=='DTW') and (destination == 'DTW' or destination == 'SEA'):
        distance = 1927.0
    elif (origin == 'MSP' or origin=='ATL') and (destination == 'ATL' or destination == 'MSP'):
        distance = 907.0
#     print(distance)
    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0,
              'DISTANCE': distance}]
    
    return model.predict_proba(pd.DataFrame(input))[0][0]

