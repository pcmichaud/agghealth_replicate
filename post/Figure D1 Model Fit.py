#!/usr/bin/env python
# coding: utf-8

# # Model fit

# In[ ]:





# In[6]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
import sys
from scipy.stats import spearmanr
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from model import micro, macro, params
from model import distributions as dist


# ## Model parameters

# In[7]:


countries = ['us','de','dk','fr','it','nl','se','sp']
df = pd.read_pickle('../estimation/output/params_ref_us.pkl')
pars = df.loc[:,'value'].to_frame()
pars.columns = ['us']
for c in countries:
	df = pd.read_pickle('../estimation/output/params_ref_'+c+'.pkl')
	pars[c] = df.loc[:,'value']
pars


# # Moments from Data

# In[8]:


moms_h = pd.read_pickle('moments/means_health.pkl')
moms_m = pd.read_pickle('moments/means_macro.pkl').transpose()
moms = moms_m.merge(moms_h,left_index=True,right_index=True,how='left')
moms = moms.rename({'gg':'trans_fromgood','gb':'trans_frombad','g_q2':'grad2','g_q3':'grad3','g_q4':'grad4'},axis=0)
moms


# In[9]:


for c in countries:
	pi= np.zeros((2,2))
	pi[0,0] = moms.loc[c.upper(),'gg']
	pi[0,1] = 1.0 - pi[0,0]
	pi[1,0] = moms.loc[c.upper(),'gb']
	pi[1,1] = 1.0 - pi[1,0]
	pi   = np.linalg.matrix_power(pi, 1000)
	moms.loc[c.upper(),'h'] = pi[0,0]
moms


# In[ ]:


moms.to_pickle('output/moms.pkl')


# # General Equilibrium

# In[10]:


def eq(co,pars):
    # estimated parameters
    p = pars.loc[:,co]
    itax = True
    theta = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                          phi=p['phi'],psi=p['psi'],delta_h1=p['delta_h1'],
                          delta_h2=p['delta_h2'],eta=0.0,tfp=p['tfp'],price=p['price'])
    # option for the numerical solution
    ne = 10
    m  = 2.5
    if co == 'us':
        op = params.settings(ne=ne,nk=30,maxk=190.0,curv=0.5,nprocs=48)
    else:
        op = params.settings(ne=ne,nk=30,maxk=150.0,curv=0.5,nprocs=48)
    inc = params.incprocess(country=co)
    inc.tauchen(ne=ne,m=m)
    aux = params.auxpars(country=co)
    #Decision rules
    csumers = micro.bellman(options=op,flex=theta,aux=aux,inc=inc,rent=5.6e-2)
    csumers.compute_cash()
    csumers.itervalue()
    # distribution
    stats = dist.stationary(dp=csumers,nk=100)
    stats.blowup()
    stats.compute()
    # general equilibrium
    eq = macro.equilibrium(stats=stats,taxes=itax,rent=True)
    eq.solve()
    aggs = eq.aggregates()
    hlth = eq.healthreport()
    s    = p['price']*aggs.M/aggs.Y
    coy  = (aggs.C + p['price']*aggs.M)/aggs.Y
    h    = hlth.pH
    tbad = hlth.pTransBad
    tgood = hlth.pTransGood
    g    = hlth.gradient
    outcomes = [coy,s,tbad,tgood,h,g[0],g[1],g[2],aggs.Y]
    names = ['cshare','mshare','gb','gg','h','g_q2','g_q3','g_q4','tfp']
    return pd.Series(index=names,data=outcomes)


# ## Simulated Moments

# In[11]:

run=False

if run:
    countries = ['us','de','dk','fr','it','nl','se','sp']
    moms_sim = pd.DataFrame(index=['cshare','mshare','gb','gg','h','g_q2','g_q3','g_q4','tfp'],columns=countries)
    for c in countries:
        print('doing country ',c)
        ms = eq(c,pars)
        moms_sim.loc[:,c] = ms


# In[ ]:

if run:
    moms_sim.loc['tfp',:] = moms_sim.loc['tfp',:]/moms_sim.loc['tfp','us']
    moms_sim = moms_sim.transpose()
    moms_sim.index = [c.upper() for c in moms_sim.index]
    moms_sim.to_pickle('output/moms_sim.pkl')
    moms_sim


# ## Figures of the fit
#

# In[ ]:


moms_sim = pd.read_pickle('output/moms_sim.pkl')
moms_sim.index = [c.upper() for c in moms_sim.index]
moms = pd.read_pickle('output/moms.pkl')

moms = moms.sort_index()
moms_sim = moms_sim.sort_index()

print(moms_sim)
print(moms)

# In[ ]:


def fit_moment(name,moms,moms_sim,show=False):
    max_m = 1.05*max(moms[name].max(),moms_sim[name].max())
    min_m = 0.95*min(moms[name].min(),moms_sim[name].min())
    plt.figure()
    plt.scatter(moms[name],moms_sim[name],facecolors='none',edgecolors='b',s=500.0)
    plt.plot(moms[name],moms[name])
    plt.xlabel('data')
    plt.xlim([min_m,max_m])
    plt.ylim([min_m,max_m])
    plt.ylabel('simulated data')
    for c in countries:
            plt.annotate(c,xy=(moms.loc[c.upper(),name], moms_sim.loc[c.upper(),name]),
            horizontalalignment='center', verticalalignment='center')
    plt.savefig('../figures/fig_d1_'+name+'.eps',dpi=1200)
    print('figure ','../figures/fig_d1_'+name+'.eps',' saved...')
    if show:
        plt.show()
    return


# In[ ]:


for m in ['mshare','tfp','gg','gb','g_q2','g_q3','g_q4','h']:
	fit_moment(m,moms,moms_sim)
	res_spear_rho, res_spear_pv = spearmanr(moms_sim[m], moms[m])
	print('moment :', m, ' ,spearman = ',[res_spear_rho, res_spear_pv])


# # Additional information

# ## EU - US Differences

# In[13]:


# pop size 2005
pop_co = ['dk','fr','de','it','nl','sp','se','us']
pops = [5149,61181,82469,58607,16319,43662,9029,295516]
pop = dict(zip(pop_co,pops))
countries_eu = [c for c in countries if c!='us']
for c in countries:
	pars.loc['pop',c] = pop[c]
price_eu = pars.loc['price',countries_eu].mean()
tfp_eu = pars.loc['tfp',countries_eu].mean()
print('- unweighted (price,tfp) = ', (price_eu,tfp_eu))
wprice_eu = (pars.loc['price',countries_eu]*pars.loc['pop',countries_eu]).sum()/pars.loc['pop',countries_eu].sum()
wtfp_eu = (pars.loc['tfp',countries_eu]*pars.loc['pop',countries_eu]).sum()/pars.loc['pop',countries_eu].sum()
print('- weighted (price,tfp) = ', (wprice_eu,wtfp_eu))


# ## TFP comparisons (Groningen vs. Banque France)

# ### TFP gaps

# In[15]:


""" from pandas import read_excel
file_name = 'xlsx_files/TFP_data.xlsx' # name of your excel file
my_sheet  = 'data2'    # sheet name
data_tfp_ts   = read_excel(file_name, sheet_name = my_sheet) """


# In[16]:


""" print("Groningen")
res_spear_rho, res_spear_pv = spearmanr(pars.loc['tfp',:], data_tfp_ts.loc[:,'Groningen 1997'])
print([res_spear_rho, res_spear_pv])
print("Banque de France")
res_spear_rho, res_spear_pv = spearmanr(pars.loc['tfp',:], data_tfp_ts.loc[:,'Banque de France'])
print([res_spear_rho, res_spear_pv]) """


# In[17]:


""" plt.figure()
plt.scatter(pars.loc['tfp',:],data_tfp_ts.loc[:,'Groningen 1997'],facecolors='none', edgecolors='b',s=500.0)
plt.scatter(pars.loc['tfp',:],data_tfp_ts.loc[:,'Banque de France'],facecolors='none', edgecolors='g',s=500.0)
plt.xlim(0.5,1.3)
plt.ylim(0.75,1.05)
plt.xlabel('Our estimates')
plt.ylabel('External information')
for x,y,z in zip(pars.loc['tfp',:],data_tfp_ts.loc[:,'Groningen 1997'],countries):
    plt.annotate(z,xy=(x, y),horizontalalignment='center', verticalalignment='center')
for x,y,z in zip(pars.loc['tfp',:],data_tfp_ts.loc[:,'Banque de France'],countries):
    plt.annotate(z,xy=(x, y),horizontalalignment='center', verticalalignment='center')
#plt.savefig('../figures_JPE/tfp_vs_data.eps',dpi=600) """

