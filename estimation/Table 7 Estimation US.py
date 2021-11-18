#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from model import micro, macro, params, distributions, calibrate
from importlib import reload
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# # Moments

# For the U.S., we use 7 moments, the consumption and health GDP shares, the transition rates from good and bad health to good health and the gradient (2, 3 and 4 relative to 1). We setup a dataframe containing the values along with the standard deviations of the moments.

# In[2]:


moms_h = pd.read_pickle('moments/means_health.pkl')
moms_m = pd.read_pickle('moments/means_macro.pkl').transpose()
moms = moms_m.merge(moms_h,left_index=True,right_index=True,how='left')
moms = moms.loc['US',:]
moms = moms.rename({'gg':'trans_fromgood','gb':'trans_frombad','g_q2':'grad2','g_q3':'grad3','g_q4':'grad4'},axis=0)
moms = moms[['cshare','mshare','trans_fromgood','trans_frombad','grad2','grad3','grad4']]
print(moms)


# In[3]:


sd_h = pd.read_pickle('moments/stds_health.pkl')
sd_m = pd.read_pickle('moments/stds_macro.pkl').transpose()
sd = sd_m.merge(sd_h,left_index=True,right_index=True,how='left')
sd = sd.loc['US',:]
sd = sd.rename({'gg':'trans_fromgood','gb':'trans_frombad','g_q2':'grad2','g_q3':'grad3','g_q4':'grad4'},axis=0)
sd = sd[['cshare','mshare','trans_fromgood','trans_frombad','grad2','grad3','grad4']]
print(sd)


# In[4]:


moms = moms.to_frame().merge(sd.to_frame(),left_index=True,right_index=True)
moms.columns = ['mean','sd']
print(moms)


# # Initial Parameters

# For the U.S., we estimate risk aversion, the health benefit $\phi$, the productivity of health $\alpha_0$ (which is $\psi$ in the code) and finally the health intercepts $\alpha_{11}$ and $\alpha_{10}$ which are $\delta_{h1}$ and $\delta_{h2}$ in the code.
#
# If need to start away from solution to see that we converge, set this switch to true:

# In[5]:


far = False


# In[6]:


if far:
	guess_us = [ 2.0,  0.2,  0.3, -0.5,  2]
else :
	guess_us = [ 2.087,  0.2703,  0.1344,
		-0.987,  3.344]

# In[7]:


ipars = calibrate.initpars(params.flexpars('us'))
ipars.fix('beta',0.97)
ipars.fix('price',1)
ipars.fix('tfp',1)
ipars.free('sigma',   guess_us[0])
ipars.free('phi',     guess_us[1])
ipars.free('psi',     guess_us[2])
ipars.free('delta_h1',guess_us[3])
ipars.free('delta_h2',guess_us[4])


# In[8]:


ipars.print()


# # Estimation

# In[9]:


prob = calibrate.msm('us',initpar=ipars,nprocs=48,ge=True)
prob.set_moments(moms)


# In[10]:


prob.estimate(maxeval=100000)


# # Standard errors

# In[11]:


prob.covar()


# # Table 7 in Paper

# In[12]:


par_labels = ['$\\sigma$','$\\phi$','$\\alpha_0$']
stat_labels = ['estimate','se']
table = pd.DataFrame(index=stat_labels,columns=par_labels)
table.loc['estimate','$\\sigma$'] = prob.opt_theta[0]
table.loc['estimate','$\\phi$'] = prob.opt_theta[1]
table.loc['estimate','$\\alpha_0$'] = prob.opt_theta[2]
table.loc['se','$\\sigma$'] = prob.se[0]
table.loc['se','$\\phi$'] = prob.se[1]
table.loc['se','$\\alpha_0$'] = prob.se[2]
table = table.round(3)
print(table)

# In[13]:


table.to_latex('../tables/table_7_common_estimates.tex',escape=False)


# # Storing Parameters for other uses

# In[14]:


l = [(p.name,p.value,p.ifree) for p in prob.initpar.pars]
df = pd.DataFrame.from_records(l)
df.columns = ['name','value','free']
df['se'] = np.nan
df.set_index('name',inplace=True)
df.loc['sigma','se'] = prob.se[0]
df.loc['phi','se'] = prob.se[1]
df.loc['psi','se'] = prob.se[2]
df.loc['delta_h1','se'] = prob.se[3]
df.loc['delta_h2','se'] = prob.se[4]
print(df)


# In[15]:


df.to_pickle('output/params_ref_us.pkl')

