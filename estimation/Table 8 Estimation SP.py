#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


co = 'sp'


# # Moments

# For European countries, we use 7 moments, the health GDP share, the transition rates from good and bad health to good health, the gradient (2, 3 and 4 relative to 1) and the GDP gap (with US). We setup a dataframe containing the values along with the standard deviations of the moments.

# In[17]:


moms_h = pd.read_pickle('moments/means_health.pkl')
moms_m = pd.read_pickle('moments/means_macro.pkl').transpose()
moms = moms_m.merge(moms_h,left_index=True,right_index=True,how='left')
moms = moms.loc[co.upper(),:]
moms = moms.rename({'gg':'trans_fromgood','gb':'trans_frombad','g_q2':'grad2','g_q3':'grad3','g_q4':'grad4'},axis=0)
moms = moms[['mshare','trans_fromgood','trans_frombad','grad2','grad3','grad4','tfp']]
moms


# In[18]:


sd_h = pd.read_pickle('moments/stds_health.pkl')
sd_m = pd.read_pickle('moments/stds_macro.pkl').transpose()
sd = sd_m.merge(sd_h,left_index=True,right_index=True,how='left')
sd = sd.loc[co.upper(),:]
sd = sd.rename({'gg':'trans_fromgood','gb':'trans_frombad','g_q2':'grad2','g_q3':'grad3','g_q4':'grad4'},axis=0)
sd = sd[['mshare','trans_fromgood','trans_frombad','grad2','grad3','grad4','tfp']]
sd


# In[19]:


moms = moms.to_frame().merge(sd.to_frame(),left_index=True,right_index=True)
moms.columns = ['mean','sd']
moms


# # Initial Parameters

# For European countries, we fix risk aversion, the health benefit $\phi$, the productivity of health $\alpha_0$ (which is $\psi$ in the code) to their values found for the U.S. We then estimate the health intercepts $\alpha_{11}$ and $\alpha_{10}$ which are $\delta_{h1}$ and $\delta_{h2}$ in the code along with the price and tfp gap.
#
# If need to start away from solution to see that we converge, set this switch to true:

# In[20]:


far = False


# In[21]:


pars_us = pd.read_pickle('output/params_ref_us.pkl')


# In[22]:


if far:
	guess = [ -.1, 4, 0.8, 0.7] 
else :
	guess = [ 0.002353013528025841953617014468, 3.434494399064982328440009951009,
		0.795094825596019183500118288066, 0.642583925569128244781325065560]


# In[23]:


ipars = calibrate.initpars(params.flexpars(co))
ipars.fix('sigma',pars_us.loc['sigma','value'])
ipars.fix('phi',  pars_us.loc['phi','value'])
ipars.fix('psi',  pars_us.loc['psi','value'])
ipars.fix('beta', 0.97)
ipars.free('delta_h1',guess[0])
ipars.free('delta_h2',guess[1])
ipars.free('tfp',     guess[2])
ipars.free('price',   guess[3])


# In[24]:


ipars.print()


# # Estimation

# In[25]:


prob = calibrate.msm(co,initpar=ipars,nprocs=36,ge=True)
prob.set_moments(moms)


# In[26]:


prob.estimate(maxeval=-1)


# # Standard errors

# In[27]:


prob.covar()


# # Storing Parameters for other uses

# In[28]:


l = [(p.name,p.value,p.ifree) for p in prob.initpar.pars]
df = pd.DataFrame.from_records(l)
df.columns = ['name','value','free']
df['se'] = np.nan
df.set_index('name',inplace=True)
df.loc['delta_h1','se'] = prob.se[0]
df.loc['delta_h2','se'] = prob.se[1]
df.loc['tfp','se'] = prob.se[2]
df.loc['price','se'] = prob.se[3]
df


# In[29]:


df.to_pickle('output/params_ref_sp.pkl')


# In[ ]:




