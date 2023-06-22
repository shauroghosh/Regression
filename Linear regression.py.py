#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = pd.read_csv("D:/Regression/1.01.+Simple+linear+regression.csv")


# In[11]:


data


# In[13]:


data.decribe()
data.describe()


# In[14]:


data


# In[15]:


data.describe()


# In[16]:


y=data['GPA']
x1=data['SAT']


# In[17]:


plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[18]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[20]:


plt.scatter(x1,y)
yhat=0.0017*x1 + 0.275
fig=plt.plot(x1,yhat,lw=4,c='orange',label ='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[ ]:




