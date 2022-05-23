#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[31]:


data = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/gender_submission.csv')
data.head()


# In[32]:


train = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/train.csv')
train.head()


# In[33]:


test = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/test.csv')
test.head()


# In[34]:


print(train.groupby('Survived').size())


# In[35]:


train.shape


# In[36]:


train.isnull().sum()


# In[37]:


train.info()


# In[38]:


train['Embarked'].unique()


# In[39]:


train.columns


# In[40]:


print(train.Name.str.extract('([A-Za-z]+)\.'))


# In[41]:


train['Age'][train['Age'].isnull()]


# In[42]:


train_Age = train['Age'].replace(np.nan, 0)
Age_mean = train_Age.sum() / 714
Age_mean


# In[43]:


train['Cabin'].isnull().sum()


# In[44]:


train['Cabin'].unique()


# In[45]:


train['Fare'].isnull()


# In[46]:


# 데이터 전처리  https://wikidocs.net/75069
# 타이타닉 데이터 설명  https://right1203.tistory.com/8

#데이터 전처리
def feature_engineering(df):
    #Sex : female=0 male=1
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
    
    #Embarked : C=0, Q=1, S=2
    df.Embarked.fillna('S', inplace=True)
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q':1, 'S':2})
    
    #Name -> Title : Master = 0, Miss=1, Mr = 2, Mrs = 3, Other = 4
    #df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
    #df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 
                                       #'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
    #df['Title'] = df['Title'].replace('Mlle', 'Miss')
    #df['Title'] = df['Title'].replace('Mme', 'Mrs')
    #df['Title'] = df['Title'].replace('Ms', 'Miss')
    #df['Title'] = df['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})
    
    # Age
    df['Age'] = df['Age'].replace(np.nan, Age_mean)
    
    # Cabin
    df.Cabin.fillna('N', inplace=True)
    df['CabinCategory'] = df['Cabin'].str.slice(start=0, stop=1)
    df['CabinCategory'] = df['CabinCategory'].map({'N': 0, 'C': 1, 'B':2, 'D':3, 'E':4,
                                                  'A':5, 'F':6, 'G':7, 'T':8})
    
    #Fare
    df.Fare.fillna(0, inplace=True)
    df['FareCategory'] = pd.qcut(df.Fare, 8, labels=range(1, 9))
    df.FareCategory = df.FareCategory.astype(int)
    
    #SibSp, Parch
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['Family'] > 4, 'Family'] = 5
    df['IsAlone'] = 1
    df.loc[df['Family'] > 1, 'IsAlone'] = 0
    
    #Ticket
    df['TicketCategory'] = df.Ticket.str.split()
    df['TicketCategory'] = [i[-1][0] for i in df['TicketCategory']]
    df['TicketCategory'] = df['TicketCategory'].replace(['8', '9', 'L'], '8')
    df['TicketCategory'] = pd.factorize(df['TicketCategory'])[0] + 1
    
    df.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare', 'Name', 'Age', 'SibSp', 'Parch'], axis=1, inplace=True)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)

train.info()
test.info()


# In[47]:


x_train = np.asarray(train.drop('Survived', 1))
y_train = np.asarray(train['Survived'])
x_test = np.asarray(test)


# In[87]:


features = [x for i, x in enumerate(train.columns) if i != 0]
features


# In[49]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit


# In[50]:


# Hyperparameter 조합 정의
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [2, 3, 4, 5],
              'min_samples_leaf': [1, 20, 100],
              'learning_rate': [0.01, 0.02, 0.05],
              'loss': ['ls']}


# In[62]:


import os
import locale
os.environ['PYTHONIOENCODING'] = 'utf-8'
scriptLocale = locale.setlocale(category=locale.LC_ALL, locale='en_GB.UTF-8')


# In[71]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))


# In[69]:


tree.plot_tree(clf)
[...]


# In[88]:


def plot_feature_importances(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(clf)

# Sex, Pclass, CabinCategory, Family, FareCategory가 중요하게 작용함을 알 수 있다


# In[89]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier

# n_estimators = 100 이 디폴트값
rf = RandomForestClassifier(n_estimators=100, random_state=0)
# fit 함수 통해서 training시킴
rf.fit(x_train, y_train)
#training set과 test set에 대해서 성능 재본다
print("Accuracy on training set: {:.3f}".format(rf.score(x_train, y_train)))
#성능도 single과 똑같음


# In[90]:


# max_depth = 3 --> 정확도 더 떨어짐
rf1 = RandomForestClassifier(max_features=2, max_depth=3, n_estimators=100, random_state=0)
rf1.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(x_train, y_train)))


# In[91]:


plot_feature_importances(rf)


# In[92]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb.score(x_train, y_train)))


# In[93]:


# max_depth = 3
gb1 = GradientBoostingClassifier(random_state=0, max_depth=2) # by deafult 3 / max_depth 디폴트 3이지만, 2로 줄여서 overfitting 막음
gb1.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(gb1.score(x_train, y_train)))


# In[94]:


plot_feature_importances(gb)


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




