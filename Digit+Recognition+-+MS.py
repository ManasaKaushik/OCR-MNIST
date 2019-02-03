
# coding: utf-8

# In[1]:


import numpy as np


# In[6]:


import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[9]:


data=pd.read_csv("desktop/train.csv").as_matrix()
dtc=DecisionTreeClassifier()
lsv=LinearSVC()
rfc=RandomForestClassifier()
gnb=GaussianNB()


# In[12]:


print(data)


# In[11]:


xtrain=data[0:21000,1:]
train_label=data[0:21000,0]


# In[18]:


dtc.fit(xtrain,train_label)
lsv.fit(xtrain,train_label)
rfc.fit(xtrain,train_label)
gnb.fit(xtrain,train_label)


# In[19]:


xtest=data[21000:,1:]
test_label=data[21000:,0]


# In[20]:


d=xtest[6]
d.shape=(28,28)


# In[16]:


pt.imshow(255-d,cmap='gray')
#print(dtc.predict( [xtest[6]] ))
#pt.show()


# In[26]:


p=dtc.predict(xtest)
q=lsv.predict(xtest)
r=rfc.predict(xtest)
s=gnb.predict(xtest)
cp=0
cq=0
cr=0
cs=0
for i in range(0,21000):
    cp+=1 if p[i]==test_label[i] else 0
    cq+=1 if q[i]==test_label[i] else 0
    cr+=1 if r[i]==test_label[i] else 0
    cs+=1 if s[i]==test_label[i] else 0
a=(cp/21000.0)
b=(cq/21000.0)
c=(cr/21000.0)
d=(cs/21000.0)
print("Decision Tree Accuracy:",a*100)
print("Linear Support Vector Machine Accuracy:",b*100)
print("Random Forest Classifier Accuracy:",c*100)
print("Gaussian Naive Bayes Accuracy:",d*100)

