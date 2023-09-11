#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[5]:


import seaborn as sns


# In[9]:


pip install pandas


# In[7]:


pip install numpy


# In[2]:


from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[150,0]]))


# In[ ]:




