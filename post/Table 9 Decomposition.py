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





# In[ ]:




