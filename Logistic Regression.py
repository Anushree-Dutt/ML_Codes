#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd


# In[26]:


df = pd.read_csv('exp4.csv')
df.head()


# In[27]:


df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()


# In[28]:


inputs = df.drop('Survived', axis='columns')
inputs.head()


# In[29]:


inputs.Sex = inputs.Sex.map({'male':1, 'female':2})
inputs


# In[30]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs


# In[31]:


target = df.Survived
target.head()


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)
print(x_train)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model = LogisticRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


accu = accuracy_score(y_test, y_pred)
print('Accuracy = ', accu)


# In[38]:


print(x_test)
print(y_test)


# In[39]:


print(model.predict([[3, 1, 29.699118, 21.6792]]))

