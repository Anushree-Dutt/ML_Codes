#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd


# In[48]:


df = pd.read_csv('exp4.csv')
df.head()


# In[49]:


df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()


# In[50]:


inputs = df.drop('Survived', axis='columns')
inputs.head()


# In[51]:


inputs.Sex = inputs.Sex.map({'male':1, 'female':2})
inputs.head()


# In[52]:


target = df.Survived
target.head()


# In[53]:


from sklearn.preprocessing import StandardScaler


# In[54]:


scaler = StandardScaler()
scaler.fit(inputs)
final_inputs = scaler.transform(inputs)
print(final_inputs)


# In[55]:


from sklearn.decomposition import PCA


# In[56]:


pca = PCA(n_components = 2)
pca.fit(final_inputs)
x = pca.transform(final_inputs)


# In[57]:


print('Eigen Vectors = \n', pca.components_)
print('Eigen Values = \n', pca.explained_variance_)

# Transfromed 2D dataset
print('Transfromed 2D dataset = \n', x)


# In[58]:


pca = PCA(n_components = 1)
pca.fit(final_inputs)
y = pca.transform(final_inputs)


# In[59]:


print('Eigen Vectors = \n', pca.components_)
print('Eigen Values = \n', pca.explained_variance_)

# Transfromed 1D dataset
print('Transfromed 1D dataset = \n', y)


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


def split_df(inputs, target):
    x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.3)
    return x_train, x_test, y_train, y_test


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


def accuracy(x_train, x_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


# In[64]:


import numpy as np


# In[65]:


df_x = pd.DataFrame(x)
df_x.head()


# In[66]:


df_y = pd.DataFrame(y)
df_y.head()


# In[67]:


x_train, x_test, y_train, y_test = split_df(inputs,target)
acc_1 = accuracy(x_train, x_test, y_train, y_test)
print("Original Dataset Accuracy:", acc_1)


# In[68]:


x_train, x_test, y_train, y_test = split_df(df_x,target)
acc_2 = accuracy(x_train, x_test, y_train, y_test)
print("2D Dataset Accuracy:",acc_2)


# In[69]:


x_train, x_test, y_train, y_test = split_df(df_y,target)
acc_3 = accuracy(x_train, x_test, y_train, y_test)
print("1D Dataset Accuracy:",acc_3)

