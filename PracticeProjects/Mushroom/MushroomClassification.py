#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[8]:


data_set=pd.read_csv('D:\FilpRobo\mushrooms.csv')


# In[9]:


data_set


# # Preprocessing

# In[15]:


mapping=list()
encoder=LabelEncoder()
for column in range(len(data_set.columns)):
    data_set[data_set.columns[column]]=encoder.fit_transform(data_set[data_set.columns[column]])
    mapping_dict={index:label for index,label in enumerate(encoder.classes_)}
    mapping.append(mapping_dict)


# In[16]:


mapping


# In[22]:


y=data_set['class']
X=data_set.drop('class',axis=1)


# In[23]:


X


# In[24]:


scaler=StandardScaler()
X=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


# In[25]:


X


# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)


# #ModelSelection

# In[29]:


ModelLR=LogisticRegression()
Modelsvm=SVC(C=1.0,kernel='rbf')
ModelML=MLPClassifier(hidden_layer_sizes=(128, 128))


# #Training

# In[30]:


np.sum(y)/len(y)


# In[31]:


ModelLR.fit(X_train,y_train)
Modelsvm.fit(X_train,y_train)
ModelML.fit(X_train,y_train)


# In[33]:


print(f"Logistic Regression: {ModelLR.score(X_test,y_test)}")
print(f"SupportVectorMachine: {Modelsvm.score(X_test,y_test)}")
print(f"Neural Network: {ModelML.score(X_test,y_test)}")


# # Plots

# In[34]:


X_test.shape


# In[39]:


correlations=data_set.corr()
sns.heatmap(correlations)


# # There is no single column can determined the ediblity. Set of all columns tend to determine the edibility

# In[ ]:




