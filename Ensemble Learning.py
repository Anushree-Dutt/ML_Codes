#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd


# In[65]:


df = pd. read_csv('exp4.csv')
df.head()


# In[66]:


df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
df.head()


# In[67]:


inputs = df.drop('Survived', axis='columns')
inputs.head()


# In[68]:


final_inputs = pd.get_dummies(inputs, columns=['Sex'])
final_inputs.head()


# In[69]:


final_inputs.Age = final_inputs.Age.fillna(inputs.Age.mean())
final_inputs


# In[70]:


target = df.Survived
target.head()


# In[71]:


from sklearn import preprocessing


# In[72]:


d = preprocessing.normalize(final_inputs)
df_final = pd.DataFrame(d, columns=["Pclass","Age","Fare","Sex_female","Sex_male"])
df_final.head()


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(final_inputs, target, test_size=0.3)


# In[75]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from vecstack import stacking
from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt


# In[76]:


# Stacking

# import numpy as np
# y_train = np.array(y_train)

model1 = LogisticRegression()
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()

all_models = [model1, model2, model3]

s_train, s_test = stacking(all_models, x_train, y_train, x_test, regression=True, random_state = None)

final_model = model1

final_model.fit(s_train, y_train)
pred = final_model.predict(s_test)

print("Root mean square error = ", sqrt(mean_squared_error(y_test, pred)))
print("Accuracy = ", accuracy_score(y_test, pred))


# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


# Bagging
clf = RandomForestClassifier()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print("Root mean square error = ", sqrt(mean_squared_error(y_test, pred)))
print("Accuracy = ", accuracy_score(y_test, pred))


# In[79]:


from xgboost import XGBClassifier


# In[80]:


# Boosting
clf = XGBClassifier()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print("Root mean square error = ", sqrt(mean_squared_error(y_test, pred)))
print("Accuracy = ", accuracy_score(y_test, pred))

