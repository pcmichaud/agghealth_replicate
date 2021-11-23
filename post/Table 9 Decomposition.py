#!/usr/bin/env python
# coding: utf-8

# # Decomposition of Effects
#
# This notebook aims to decompose differences in the GDP share of health spending and health levels across countries into health service wedges (prices), efficiency wedges (TFP) and health risk wedges ($\delta_{10},\delta_{11}$).
# Scenarios are run in general equilibrium and $s$ and $h$ differences are aggregated within Europe. Intermediate results are saved as pickle dataframes and the main table is outputted to tex.  The output is Table 9 in the paper.

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from importlib import reload
import pandas as pd
from itertools import product
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from model import micro,macro,params
from model import distributions as dist


# Creating Table

countries = ['us','fr','sp','it','dk','de','nl','se']
regions = ['us','eu','$\\Delta$']
stats = ['s','h','g']
scenarios = ['baseline','price','efficiency','health risks']

cols = [stats,regions]
tups = list(product(*cols))
index = pd.MultiIndex.from_tuples(tups, names=['stats', 'regions'])
table = pd.DataFrame(index=scenarios,columns=index)
print(table)

# loading results (need run the table9-scn.py files to get results)
eu_countries = [c for c in countries if c!='us']

results_ref = pd.read_pickle('output/table9-ref.pkl')
results_prices = pd.read_pickle('output/table9-prices.pkl')
results_tfp = pd.read_pickle('output/table9-tfp.pkl')
results_risks = pd.read_pickle('output/table9-risks.pkl')

for s in stats:
    table.loc['baseline',(s,'us')] = results_ref.loc[s,'us']
    table.loc['baseline',(s,'eu')] = results_ref.loc[s,eu_countries].mean()
    table.loc['price',(s,'us')] = results_prices.loc[s,'us']
    table.loc['price',(s,'eu')] = results_prices.loc[s,eu_countries].mean()
    table.loc['efficiency',(s,'us')] = results_tfp.loc[s,'us']
    table.loc['efficiency',(s,'eu')] = results_tfp.loc[s,eu_countries].mean()
    table.loc['health risks',(s,'us')] = results_risks.loc[s,'us']
    table.loc['health risks',(s,'eu')] = results_risks.loc[s,eu_countries].mean()
    table.loc[:,(s,'$\\Delta$')] = table.loc[:,(s,'us')] - table.loc[:,(s,'eu')]
print(table)

table.round(3).to_latex('../tables/table_9_decomposition.tex')















# # Table of results where the benchmark and the price scenarios are not with the same grids than the others scenarios

# In[ ]:


""" # Results for benchmark and price scenarios
# s
sb = [0.140133, 0.10057,  0.077059, 0.086972, 0.087288,  0.09904,   0.0889796, 0.0903872]
sp = [0.110341, 0.126691, 0.087393, 0.10693,  0.0730481, 0.0889361, 0.106651,  0.0733584]
# h
hb = [0.896541 , 0.9597 , 0.937642 , 0.925261 , 0.973665 , 0.95948 , 0.965963 , 0.967705]
hp = [0.925708 , 0.950858 , 0.929576 , 0.90776 , 0.978669 , 0.965688 , 0.962528 , 0.975766]
# c
cb = [0.679479 , 0.717629 , 0.737936 , 0.669748 , 0.730785 , 0.716155 , 0.713161 , 0.673503]
cp = [0.677385 , 0.755819 , 0.778857 , 0.738478 , 0.775511 , 0.746577 , 0.775522 , 0.712928]
# g
gb = [1.27547 , 1.06596 , 1.09368 , 1.08953 , 1.02656 , 1.06441 , 1.05499 , 1.01917]
gp = [1.1825 , 1.08826 , 1.13641 , 1.1457 , 1.01842 , 1.05508 , 1.069 , 1.01558] """


# In[ ]:


""" results_s.loc['baseline',:] = sb
results_s.loc['price',:]    = sp

results_h.loc['baseline',:] = hb
results_h.loc['price',:]    = hp

results_c.loc['baseline',:] = cb
results_c.loc['price',:]    = cp

results_g.loc['baseline',:] = gb
results_g.loc['price',:]    = gp """




# ## Saving results

# In[20]:


""" results_s.to_pickle('../output_JPE/decomp_share.pkl')
results_h.to_pickle('../output_JPE/decomp_hlth.pkl')
results_c.to_pickle('../output_JPE/decomp_csy.pkl')
results_g.to_pickle('../output_JPE/decomp_gradient.pkl') """


# ## Loading (if not re-running)

# In[19]:


""" results_s = pd.read_pickle('../output_JPE/decomp_share.pkl')
results_h = pd.read_pickle('../output_JPE/decomp_hlth.pkl')
results_c = pd.read_pickle('../output_JPE/decomp_csy.pkl')
results_g = pd.read_pickle('../output_JPE/decomp_gradient.pkl') """


# ## Fill in Results Table

# In[11]:


# In[22]:


""" for scn in scenarios:
    table.loc[scn,('s','us')] = results_s.loc[scn,'us']
    table.loc[scn,('s','eu')] = np.mean(results_s.loc[scn,eu_countries])
    table.loc[scn,('s','$\\Delta$')] = table.loc[scn,('s','us')]-table.loc[scn,('s','eu')]
    table.loc[scn,('h','us')] = results_h.loc[scn,'us']
    table.loc[scn,('h','eu')] = np.mean(results_h.loc[scn,eu_countries])
    table.loc[scn,('h','$\\Delta$')] = table.loc[scn,('h','us')]-table.loc[scn,('h','eu')]
    table.loc[scn,('g','us')] = results_g.loc[scn,'us']
    table.loc[scn,('g','eu')] = np.mean(results_g.loc[scn,eu_countries])
    table.loc[scn,('g','$\\Delta$')] = table.loc[scn,('g','us')]-table.loc[scn,('g','eu')]   """


# In[23]:


# In[24]:




# In[ ]:




