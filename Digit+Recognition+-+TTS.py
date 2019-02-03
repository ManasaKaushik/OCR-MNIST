
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:


data=pd.read_csv("desktop/train.csv").as_matrix()
dtc=DecisionTreeClassifier()
lsv=LinearSVC()
rfc=RandomForestClassifier()
gnb=GaussianNB()


# In[6]:


print(data)


# In[7]:


x=data[0:42000,1:]
y=data[0:42000,0]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)
#train_label=data[0:33600,0]


# In[8]:


dtc.fit(x_train,y_train)
lsv.fit(x_train,y_train)
rfc.fit(x_train,y_train)
gnb.fit(x_train,y_train)


# In[9]:


d=x_test[6]
d.shape=(28,28)


# In[10]:


pt.imshow(255-d,cmap='gray')
#print(dtc.predict( [xtest[6]] ))
#pt.show()


# In[19]:


p=dtc.predict(x_test)
print(accuracy_score(p,y_test))
q=lsv.predict(x_test)
print(accuracy_score(q,y_test))
r=rfc.predict(x_test)
print(accuracy_score(r,y_test))
s=gnb.predict(x_test)
print(accuracy_score(s,y_test))

