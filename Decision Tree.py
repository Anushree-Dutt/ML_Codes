#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('exp4.csv')
df.head()


# In[3]:


df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()


# In[17]:


inputs = df.drop('Survived', axis='columns')
inputs.head()


# In[5]:


inputs.Sex = inputs.Sex.map({'male':1, 'female':2})
inputs


# In[6]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs


# In[7]:


target = df.Survived
target.head()


# In[8]:


from sklearn import tree 
from sklearn.model_selection import train_test_split


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)


# In[10]:


model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[11]:


print(x_test)
print(y_test)


# In[12]:


model.predict([[2, 1, 23.000000, 13.0000]])


# In[13]:


text_tree = tree.export_text(model)
print(text_tree)


# In[14]:


from matplotlib import pyplot as plt


# In[15]:


fig = plt.figure(figsize=(200,200))
tree_ = tree.plot_tree(model, feature_names=["Pclass","Sex","Age","Fare"], class_names='Survived', fontsize=100)

