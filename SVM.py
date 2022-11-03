#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv('exp4.csv')
df.head()


# In[7]:


df.Sex = df.Sex.map({'male':1, 'female':2})
df.head()


# In[8]:


df.corr()


# In[9]:


# Features with highest correlation - Sex, Pclass, Fare

df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()


# In[10]:


inputs = df.drop('Survived', axis='columns')
inputs.head()


# In[11]:


target = df.Survived
target.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)
print(x_train)


# In[14]:


from sklearn.svm import SVC


# In[15]:


clf = SVC(kernel='linear', random_state=0)
clf.fit(x_train, y_train)


# In[16]:


clf.support_vectors_


# In[17]:


clf.n_support_


# In[18]:


clf.support_


# In[19]:


y_pred = clf.predict(x_test)
print(y_pred)


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[23]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy = ', accuracy)

positive_precision = precision_score(y_test, y_pred, pos_label=1)
negative_precision = precision_score(y_test, y_pred, pos_label=0)
print('Positive Precision = ', positive_precision)
print('Negative Precision = ', negative_precision)

recall_sensitivity = recall_score(y_test, y_pred, pos_label=1)
recall_specificity = recall_score(y_test, y_pred, pos_label=0)
print('Recall sensitivity = ', recall_sensitivity)
print('Recall specificity = ', recall_specificity)

