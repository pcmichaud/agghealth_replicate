#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from itertools import product


# In[2]:


countries = ['us','de','dk','fr','it','nl','se','sp']


# In[3]:


test = pd.read_pickle('output/params_wage_us.pkl')
test


# In[4]:


pars_select = ['delta_h1','delta_h2','price','tfp']
stat = ['par','se']
tuples = list(product(*[pars_select,stat]))
table = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples),columns=countries)
for c in countries: 
	df = pd.read_pickle('output/params_wage_'+c+'.pkl')
	for p in pars_select:
		table.loc[(p,'par'),c] = df.loc[p,'value']
		table.loc[(p,'se'),c] = df.loc[p,'se']


# In[5]:


pars_select = ['delta_h1','delta_h2','price','tfp']
labels = ['$\\alpha_{10}$','$\\alpha_{11}$','$\\frac{p}{p_{US}}$','$\\frac{A}{A_{US}}$']
maps_labels = dict(zip(pars_select,labels))
table.index = pd.MultiIndex.from_tuples(list(product(*[labels,['','se']])))


# In[6]:


for c in table.columns:
	table[c] = table[c].astype('float64')
	table[c] = table[c].round(3)
	table[c] = table[c].astype('str')
	table[c] = np.where(table[c]=='nan','-',table[c])
	for s in table.index:
		if s[1]=='se':
			if table.loc[s,c]!='-':
				fmt = "({se})"
				table.loc[s,c] = fmt.format(se=table.loc[s,c])
table.index = pd.MultiIndex.from_tuples(list(product(*[labels,['','']])))


# In[7]:


table


# In[ ]:





# In[8]:


table.to_latex('../tables/table_e2_country_specific_estimates.tex')


# ## Average US-EU differences
# 
# This number is used in counterfactual analysis. 

# In[9]:


table.index = pd.MultiIndex.from_tuples(list(product(*[labels,['p','se']])))
prices = table.loc[('$\\frac{p}{p_{US}}$','p'),:]
prices['de':].astype('float64').mean()


# In[10]:


table.index = pd.MultiIndex.from_tuples(list(product(*[labels,['p','se']])))
prices = table.loc[('$\\frac{A}{A_{US}}$','p'),:]
prices['de':].astype('float64').mean()


# In[ ]:





# In[ ]:




