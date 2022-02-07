#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


sns.set_style('darkgrid')
plt.rcParams['font.size']=15
plt.rcParams['figure.figsize']=(10,7)
plt.rcParams['figure.facecolor']='#FFE5b4'
data_set=pd.read_csv('D:\\FilpRobo\\happiness_score_dataset.csv')


# In[16]:


data_set.head()


# In[17]:


data_columns=['Country','Region','Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual']


# In[18]:


data_columns


# In[19]:


data_set=data_set[data_columns].copy()


# In[20]:


data_set


# In[22]:


data_set.rename(columns={'Country':'country','Region':'RegInd','Happiness Score':'HappyScore','Economy (GDP per Capita)':'LGDP','Health (Life Expectancy)':'HealthylifeExp','Freedom':'FreedomlifeChoices','Trust (Government Corruption)':'PerceptionCorruption','Generosity':'generosity','Dystopia Residual':'Dystopia'},inplace=True)


# In[44]:


data_set.head()


# In[67]:


data_set.isnull().sum()


# In[68]:


#happy vs GDP
#plt.recParams['figure'.'figsize']=(15,7)
plt.title('Plot HappyScore Vs GDP')
sns.scatterplot(x=data_set.HappyScore,y=data_set.LGDP,hue=data_set.RegInd);
plt.legend(loc='upper left',fontsize='10')
plt.xlabel('Happy Score')
plt.ylabel('GDP per Capita')


# In[69]:


gdp_region=data_set.groupby('RegInd')['LGDP'].sum()


# In[70]:


gdp_region


# In[71]:


gdp_region.plot.pie(autopct='%1.1f%%')
plt.title('GDP per rigion')
plt.ylabel('')


# In[72]:


tot_count=data_set.groupby('RegInd')[['country']].count()
tot_count


# In[74]:


##corelation Map
cor=data_set.corr(method="pearson")
f,ax=plt.subplots(figsize=(10,5))
sns.heatmap(cor,mask=np.zeros_like(cor,dtype=np.bool),cmap="Blues",square=True, ax=ax)


# In[81]:


##corruption in different regions
corruption=data_set.groupby('RegInd')[['PerceptionCorruption']].mean()


# In[82]:


corruption


# In[83]:


plt.rcParams['figure.figsize']=(12,8)
plt.title('Perception of Corruption')
plt.xlabel('Regions',fontsize=15)
plt.ylabel('Corruption Index',fontsize=15)
plt.xticks(rotation=30,ha='right')
plt.bar(corruption.index,corruption.PerceptionCorruption)


# In[84]:


top10=data_set.head(10)
bottom10=data_set.tail(10)
bottom10


# In[85]:


fig, axes=plt.subplots(1,2, figsize=(16,6))
plt.tight_layout(pad=2)
xlabels=top10.country
axes[0].set_title('Top 10 Countries Life Expentency')
axes[0].set_xticklabels(xlabels,rotation=45,ha='right')
sns.barplot(x=top10.country,y=top10.HealthylifeExp, ax=axes[0])
axes[0].set_xlabel('Country')
axes[0].set_ylabel('Life Expentency')

xlabels=bottom10.country
axes[1].set_title('Bottom 10 Countries Life Expentency')
axes[1].set_xticklabels(xlabels,rotation=45,ha='right')
sns.barplot(x=bottom10.country,y=bottom10.HealthylifeExp, ax=axes[1])
axes[1].set_xlabel('Country')
axes[1].set_ylabel('Life Expentency')


# In[86]:


plt.rcParams['figure.figsize']=(15,7)
sns.scatterplot(x=data_set.FreedomlifeChoices, y=data_set.HappyScore, hue=data_set.RegInd , s=200)
plt.legend(loc='upper left',fontsize='12')
plt.xlabel('Freedom of making Life choices')
plt.ylabel('Happiness Score')


# In[39]:


top10corruption=data_set.sort_values(by='PerceptionCorruption').head(10)
top10corruption


# In[87]:


plt.rcParams['figure.figsize']=(12,6)
plt.title('Countries of perception of corruption')
plt.xlabel('top10corruption',fontsize=14)
plt.ylabel('Corruption Index',fontsize=14)
plt.xticks(rotation=30, ha='right')
plt.bar(top10corruption.country,top10corruption.PerceptionCorruption)


# In[89]:


##corruption vs happiness
plt.rcParams['figure.figsize']=(15,7)
sns.scatterplot(x=data_set.HappyScore,y=data_set.PerceptionCorruption, hue=data_set.RegInd, s=200)
plt.legend(loc='lower left',fontsize='16')
plt.xlabel('Happiness Score')
plt.ylabel('Corruption')


# In[ ]:




