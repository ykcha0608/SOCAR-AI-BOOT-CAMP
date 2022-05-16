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


# In[3]:


data = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/gender_submission.csv')
data.head()


# In[4]:


train = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/train.csv')
train.head()


# In[5]:


test = pd.read_csv('C:/Users/차유경/Documents/차유경/Scocar AI Prj/data/titanic/test.csv')
test.head()


# In[6]:


print(train.groupby('Survived').size())


# In[7]:


train.shape


# In[8]:


train.isnull().sum()


# In[9]:


train.info()


# In[10]:


train['Embarked'].unique()


# In[11]:


train.columns


# In[15]:


print(train.Name.str.extract('([A-Za-z]+)\.'))


# In[19]:


train['Age'][train['Age'].isnull()]


# In[24]:


train_Age = train['Age'].replace(np.nan, 0)
Age_mean = train_Age.sum() / 714
Age_mean


# In[26]:


train['Cabin'].isnull().sum()


# In[27]:


train['Cabin'].unique()


# In[29]:


train['Fare'].isnull()


# In[ ]:


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
                                       'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
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
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


type(train)


# In[10]:


test.head()


# In[11]:


x_train = np.asarray(train.drop(['Survived', 'Name', 'Ticket'], axis=1))
y_train = np.asarray(train['Survived'])
x_test = np.asarray(train.drop(['Survived', 'Name', 'Ticket'], axis=1))


# In[12]:


x_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


# floattensor로 바꾸기
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)


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




