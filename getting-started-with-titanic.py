#!/usr/bin/env python
# coding: utf-8

# In[76]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[77]:


train_data = pd.read_csv('./train.csv')
train_data.head()


# In[78]:


train_data.info()


# In[79]:


test_data = pd.read_csv("./test.csv")
test_data.head()


# In[80]:


test_data.info()


# # Determine the Baseline:
# 

# In[81]:


train_data.Survived .value_counts()  


# In[82]:


train_data.Pclass.value_counts() 


# In[83]:


train_data.Age.value_counts()


# The Age attribute have many unique values and when we need to deal with this  column we need more preprocessing operations so, i will drop this column.

# In[84]:


train_data.Name.value_counts()


# The Name attribute have many unique values and when we need to deal with this  column we need more preprocessing operations so, i will drop this column.

# In[85]:


train_data.SibSp.value_counts()


# In[86]:


train_data.Parch.value_counts()


# In[87]:


train_data.Sex.value_counts()


# In[88]:


train_data.Ticket.value_counts()


# The Ticket attribute have many unique values and when we need to deal with this  column we need more preprocessing operations so, i will drop this column.

# In[89]:


train_data.Fare.value_counts()


# The Fare attribute have many unique values and when we need to deal with this  column we need more preprocessing operations so, i will drop this column.

# In[90]:


train_data.Cabin.value_counts()


# The Cabin attribute have many unique values and when we need to deal with this  column we need more preprocessing operations so, i will drop this column.

# In[91]:


train_data.Embarked .value_counts()
 


# we see in above, we have in attributes( Name,Ticket,Age,Fare,Cabian)-> will drop 
# 

# # And to drop these atteributes :
# 

# In[92]:


train_data.drop(['Cabin','Fare','Age','Ticket','Name'],inplace=True,axis=1)
test_data.drop(['Cabin','Fare','Age','Ticket','Name'],inplace=True,axis=1)


# # The new Baseline is:

# In[93]:


train_data.info()
test_data.info()


# we see on above, The attribute(Embarked) have null valuse so, we need to fix it:

# In[105]:


#Embarked null fix
data = [train_data, test_data]

for dataset in data:
    dataset.Embarked = dataset.Embarked.fillna('S')


# In[106]:


train_data.info()
test_data.info()


# Now we need to convert all objects and strings into numeric values:
# we have just 2 attributes (Sex and Embarked)

# In[96]:


train_data.Sex.value_counts()


# In[97]:


train_data.Embarked.value_counts()


# In[98]:


genderMap = {"male": 0, "female": 1}
embarkedMap = {"S": 0, "C": 1, "Q":2}


data = [train_data, test_data] 

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genderMap)
    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)


# In[99]:


train_data.info()
test_data.info()


# # Select X's and Y's

# In[107]:


X_train=train_data.drop(['Survived','PassengerId'], axis=1)
Y_train=train_data['Survived']


# In[108]:


X_test=test_data.drop('PassengerId', axis=1)


# # Select a model and train it:

# I will use Regression model:(i see this tutorial https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# In[112]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, Y_train)


Y_pred=clf.predict(X_test)


# In[114]:


print(Y_pred)


# # Evaluation Model:

# In[118]:


acc_logistic = round(clf.score(X_train, Y_train)*110, 2)

print (acc_logistic)


# # submit to leaderboard:

# In[119]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




