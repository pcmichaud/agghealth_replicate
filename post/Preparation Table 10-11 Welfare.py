#!/usr/bin/env python
# coding: utf-8

# # Welfare: simulations for variations in price or TFP, at GE or PE

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from importlib import reload
import pandas as pd
import pickle


# In[2]:


from model import micro,macro,params
from model import distributions as dist


# ## Model parameters

# In[3]:


df = pd.read_pickle('../estimation/output/params_ref_us.pkl')
pars = df.loc[:,'value'].to_frame()
pars.columns = ['us']
for c in countries:
    df = pd.read_pickle('../estimation/output/params_ref_'+c+'.pkl')
    pars[c] = df.loc[:,'value']
print(pars)




# ## GE benchmark

# In[4]:


countries = pars.columns
scenarios = ['pus_ge']#,'peu_ge','peu_pe','aeu_ge','aeu_pe']
outcomes  = ['m','y','c','k','n','s','csy','ksy','h','g2','g3','g4','tgood','tbad','r','w','tax','oop']
results   = pd.DataFrame(index=outcomes,columns=scenarios)


# In[5]:


p = pars.loc[:,'us']
theta = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                            phi=p['phi'],psi=p['psi'],delta_h1=p['d1'],
                            delta_h2=p['d2'],eta=0.0,tfp=p['tfp'],price=p['p'])
# option for the numerical solution
ne = 10
m  = 2.5
#op = params.settings(ne=ne,nk=30,maxk=190.0,curv=0.5,nprocs=40)
op = params.settings(ne=ne,nk=100,maxk=190.0,curv=0.5,nprocs=40)
inc = params.incprocess(country='us')
inc.tauchen(ne=ne,m=m)
aux = params.auxpars(country='us')
#Decision rules
csumers = micro.bellman(options=op,flex=theta,aux=aux,inc=inc,rent=5.6e-2)
csumers.compute_cash()
csumers.itervalue()
# distribution
stats = dist.stationary(dp=csumers,nk=500)
stats.blowup()
stats.compute()
# general equilibrium
eq = macro.equilibrium(stats=stats,taxes=False,rent=True)#,inirent=irate)
eq.solve()
aggs = eq.aggregates()
hlth = eq.healthreport()

# saving aggregate outcomes
res = [aggs.M,aggs.Y,aggs.C,aggs.K,aggs.N,p['p']*aggs.M/aggs.Y,(aggs.C+p['p']*aggs.M)/aggs.Y,
       aggs.K/aggs.Y,hlth.pH,hlth.gradient[0],hlth.gradient[1],hlth.gradient[2],hlth.pTransGood,
       hlth.pTransBad,eq.rent,eq.wage,eq.tax,aggs.M*p['p']*aux.copay]
# saving decision rules
size = stats.ne*stats.nh*stats.nk
opt = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt.loc[:,'ps'] = eq.stats.probs
for i,s in enumerate(stats.states):
    e,h,k = s
    opt.loc[i,['e','h','k']] = [e,h,k]
    opt.loc[i,'c'] = eq.stats.optc[e,h,k]
    opt.loc[i,'m'] = eq.stats.optm[e,h,k]
    opt.loc[i,'kp'] = eq.stats.optk[e,h,k]
    opt.loc[i,'v'] = eq.stats.value[e,h,k]


# In[6]:


eqs = []
results.loc[:,'pus_ge'] = res
eqs.append(eq)


# In[7]:


results.loc['r','pus_ge']


# In[8]:


results.loc['s','pus_ge']


# In[9]:


results.loc['tbad','pus_ge']


# In[ ]:





# In[5]:


p = pars.loc[:,'us']
theta = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                            phi=p['phi'],psi=p['psi'],delta_h1=p['d1'],
                            delta_h2=p['d2'],eta=0.0,tfp=p['tfp'],price=p['p'])
# option for the numerical solution
ne = 10
m  = 2.5
#op = params.settings(ne=ne,nk=30,maxk=190.0,curv=0.5,nprocs=40)
op = params.settings(ne=ne,nk=100,maxk=190.0,curv=0.5,nprocs=40)
inc = params.incprocess(country='us')
inc.tauchen(ne=ne,m=m)
aux = params.auxpars(country='us')
#Decision rules
csumers = micro.bellman(options=op,flex=theta,aux=aux,inc=inc,rent=5.6e-2)
csumers.compute_cash()
csumers.itervalue()
# distribution
stats = dist.stationary(dp=csumers,nk=500)
stats.blowup()
stats.compute()
# general equilibrium
eq = macro.equilibrium(stats=stats,taxes=False,rent=True)#,inirent=irate)
eq.solve()
aggs = eq.aggregates()
hlth = eq.healthreport()

# saving aggregate outcomes
res = [aggs.M,aggs.Y,aggs.C,aggs.K,aggs.N,p['p']*aggs.M/aggs.Y,(aggs.C+p['p']*aggs.M)/aggs.Y,
       aggs.K/aggs.Y,hlth.pH,hlth.gradient[0],hlth.gradient[1],hlth.gradient[2],hlth.pTransGood,
       hlth.pTransBad,eq.rent,eq.wage,eq.tax,aggs.M*p['p']*aux.copay]
# saving decision rules
size = stats.ne*stats.nh*stats.nk
opt = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt.loc[:,'ps'] = eq.stats.probs
for i,s in enumerate(stats.states):
    e,h,k = s
    opt.loc[i,['e','h','k']] = [e,h,k]
    opt.loc[i,'c'] = eq.stats.optc[e,h,k]
    opt.loc[i,'m'] = eq.stats.optm[e,h,k]
    opt.loc[i,'kp'] = eq.stats.optk[e,h,k]
    opt.loc[i,'v'] = eq.stats.value[e,h,k]


# In[6]:


eqs = []


# In[7]:


#scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
opt.to_pickle('../output_JPE/opt_pus_ge3.pkl')
file = open('../output_JPE/eq_pus_ge3.pkl','wb')
pickle.dump(eq,file)
file.close()
results.loc[:,'pus_ge'] = res
eqs.append(eq)


# In[8]:


results.loc['r','pus_ge']


# In[9]:


results.loc['s','pus_ge']


# In[10]:


results.loc['tbad','pus_ge']


# ## GE with EU price

# In[11]:


price_eu = pars.loc['p',[c for c in countries if c!='us']].mean()
price_eu


# In[12]:


# Change in health price
theta0 = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                            phi=p['phi'],psi=p['psi'],delta_h1=p['d1'],
                            delta_h2=p['d2'],eta=0.0,tfp=p['tfp'],price=price_eu)


# In[13]:


#Decision rules
csumers0 = micro.bellman(options=op,flex=theta0,aux=aux,inc=inc,rent=5.6e-2)
csumers0.compute_cash()
csumers0.itervalue()
# distribution
stats0 = dist.stationary(dp=csumers0,nk=500)
stats0.blowup()
stats0.compute()
# general equilibrium
eq0 = macro.equilibrium(stats=stats0,taxes=True,rent=True)#,inirent=irate)
eq0.solve()
aggs0 = eq0.aggregates()
hlth0 = eq0.healthreport()

# saving aggregate outcomes
res0 = [aggs0.M,aggs0.Y,aggs0.C,aggs0.K,aggs0.N,price_eu*aggs0.M/aggs0.Y,(aggs0.C+price_eu*aggs0.M)/aggs0.Y,
       aggs0.K/aggs0.Y,hlth0.pH,hlth0.gradient[0],hlth0.gradient[1],hlth0.gradient[2],hlth0.pTransGood,
       hlth0.pTransBad,eq0.rent,eq0.wage,eq0.tax,aggs0.M*price_eu*aux.copay]
# saving decision rules
size = stats0.ne*stats0.nh*stats0.nk
opt0 = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt0.loc[:,'ps'] = eq0.stats.probs
for i,s in enumerate(stats0.states):
    e,h,k = s
    opt0.loc[i,['e','h','k']] = [e,h,k]
    opt0.loc[i,'c']  = eq0.stats.optc[e,h,k]
    opt0.loc[i,'m']  = eq0.stats.optm[e,h,k]
    opt0.loc[i,'kp'] = eq0.stats.optk[e,h,k]
    opt0.loc[i,'v']  = eq0.stats.value[e,h,k]


# In[14]:


eqs0 = []


# In[15]:


scenarios = ['peu_ge']
results0  = pd.DataFrame(index=outcomes,columns=scenarios)


# In[16]:


#scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
opt0.to_pickle('../output_JPE/opt_peu_ge3.pkl')
file = open('../output_JPE/eq_peu_ge3.pkl','wb')
pickle.dump(eq0,file)
file.close()
results0.loc[:,'peu_ge'] = res0
eqs0.append(eq0)


# In[17]:


results0.loc['r','peu_ge']


# In[18]:


results0.loc['s','peu_ge']


# In[19]:


results0.loc['tbad','peu_ge']


# In[20]:


results.loc['r','pus_ge']-results0.loc['r','peu_ge']


# In[21]:


(results0.loc['w','peu_ge']-results.loc['w','pus_ge'])/results.loc['w','pus_ge']


# In[22]:


results.loc['tax','pus_ge']-results0.loc['tax','peu_ge']


# In[23]:


((1-results0.loc['tax','peu_ge'])*results0.loc['w','peu_ge']-(1-results.loc['tax','pus_ge'])*results.loc['w','pus_ge'])/((1-results.loc['tax','pus_ge'])*results.loc['w','pus_ge'])


# ## PE with EU price

# In[24]:


#Decision rules
csumers1 = micro.bellman(options=op,flex=theta0,aux=aux,inc=inc,rent=results.loc['r','pus_ge'],
                                                                taxrate=results.loc['tax','pus_ge'],
                                                                wage=results.loc['w','pus_ge'])
csumers1.compute_cash()
csumers1.itervalue()
# distribution
stats1 = dist.stationary(dp=csumers1,nk=500)
stats1.blowup()
stats1.compute()
# general equilibrium
eq1 = macro.equilibrium(stats=stats1,taxes=False,rent=False,inirent=results.loc['r','pus_ge'],
                                                            initax=results.loc['tax','pus_ge'])
eq1.solve()
aggs1 = eq1.aggregates()
hlth1 = eq1.healthreport()

# saving aggregate outcomes
res1 = [aggs1.M,aggs0.Y,aggs1.C,aggs1.K,aggs1.N,price_eu*aggs1.M/aggs1.Y,(aggs1.C+price_eu*aggs1.M)/aggs1.Y,
       aggs1.K/aggs0.Y,hlth1.pH,hlth1.gradient[0],hlth1.gradient[1],hlth1.gradient[2],hlth1.pTransGood,
       hlth1.pTransBad,eq1.rent,eq1.wage,eq1.tax,aggs1.M*price_eu*aux.copay]
# saving decision rules
size = stats1.ne*stats1.nh*stats1.nk
opt1 = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt1.loc[:,'ps'] = eq1.stats.probs
for i,s in enumerate(stats1.states):
    e,h,k = s
    opt1.loc[i,['e','h','k']] = [e,h,k]
    opt1.loc[i,'c']  = eq1.stats.optc[e,h,k]
    opt1.loc[i,'m']  = eq1.stats.optm[e,h,k]
    opt1.loc[i,'kp'] = eq1.stats.optk[e,h,k]
    opt1.loc[i,'v']  = eq1.stats.value[e,h,k]


# In[25]:


eqs1 = []


# In[26]:


scenarios = ['peu_pe']
results1  = pd.DataFrame(index=outcomes,columns=scenarios)


# In[27]:


#scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
opt1.to_pickle('../output_JPE/opt_peu_pe3.pkl')
file = open('../output_JPE/eq_peu_pe3.pkl','wb')
pickle.dump(eq1,file)
file.close()
results1.loc[:,'peu_pe'] = res1
eqs1.append(eq1)


# In[28]:


results1.loc['r','peu_pe']


# In[29]:


results1.loc['s','peu_pe']


# In[30]:


results1.loc['tbad','peu_pe']


# ## GE with EU TFP

# In[31]:


tfp_eu = pars.loc['tfp',[c for c in countries if c!='us']].mean()
tfp_eu


# In[32]:


theta1 = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                         phi=p['phi'],psi=p['psi'],delta_h1=p['d1'],
                         delta_h2=p['d2'],eta=0.0,tfp=tfp_eu,price=p['p'])


# In[33]:


#Decision rules
csumers2 = micro.bellman(options=op,flex=theta1,aux=aux,inc=inc,rent=5.6e-2)
csumers2.compute_cash()
csumers2.itervalue()
# distribution
stats2 = dist.stationary(dp=csumers2,nk=500)
stats2.blowup()
stats2.compute()
# general equilibrium
eq2 = macro.equilibrium(stats=stats2,taxes=True,rent=True)
eq2.solve()
aggs2 = eq2.aggregates()
hlth2 = eq2.healthreport()

# saving aggregate outcomes
res2 = [aggs2.M,aggs2.Y,aggs2.C,aggs2.K,aggs2.N,price_eu*aggs2.M/aggs2.Y,(aggs2.C+p['p']*aggs2.M)/aggs2.Y,
       aggs2.K/aggs2.Y,hlth2.pH,hlth2.gradient[0],hlth2.gradient[1],hlth2.gradient[2],hlth2.pTransGood,
       hlth2.pTransBad,eq2.rent,eq2.wage,eq2.tax,aggs2.M*p['p']*aux.copay]
# saving decision rules
size = stats2.ne*stats2.nh*stats2.nk
opt2 = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt2.loc[:,'ps'] = eq2.stats.probs
for i,s in enumerate(stats2.states):
    e,h,k = s
    opt2.loc[i,['e','h','k']] = [e,h,k]
    opt2.loc[i,'c']  = eq2.stats.optc[e,h,k]
    opt2.loc[i,'m']  = eq2.stats.optm[e,h,k]
    opt2.loc[i,'kp'] = eq2.stats.optk[e,h,k]
    opt2.loc[i,'v']  = eq2.stats.value[e,h,k]


# In[34]:


eqs2 = []


# In[35]:


scenarios = ['aeu_ge']
results2  = pd.DataFrame(index=outcomes,columns=scenarios)


# In[36]:


#scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
opt2.to_pickle('../output_JPE/opt_aeu_ge3.pkl')
file = open('../output_JPE/eq_aeu_ge3.pkl','wb')
pickle.dump(eq2,file)
file.close()
results2.loc[:,'aeu_ge'] = res2
eqs2.append(eq2)


# In[37]:


results2.loc['r','aeu_ge']


# In[38]:


results2.loc['s','aeu_ge']


# In[39]:


results2.loc['tbad','aeu_ge']


# ## PE with EU TFP

# In[40]:


# Partial equilibrium for househlods => decline of the wage
#Decision rules
csumers3 = micro.bellman(options=op,flex=theta1,aux=aux,inc=inc,rent=results.loc['r','pus_ge'],
                                                                taxrate=results.loc['tax','pus_ge'],
                                                                wage=tfp_eu*results.loc['w','pus_ge'])
csumers3.compute_cash()
csumers3.itervalue()
# distribution
stats3 = dist.stationary(dp=csumers3,nk=500)
stats3.blowup()
stats3.compute()
# general equilibrium
eq3 = macro.equilibrium(stats=stats3,taxes=False,rent=False,inirent=results.loc['r','pus_ge'],
                                                            initax=results.loc['tax','pus_ge'])
eq3.solve()
aggs3 = eq3.aggregates()
hlth3 = eq3.healthreport()

# saving aggregate outcomes
res3 = [aggs3.M,aggs0.Y,aggs3.C,aggs3.K,aggs3.N,price_eu*aggs3.M/aggs3.Y,(aggs3.C+price_eu*aggs3.M)/aggs3.Y,
       aggs3.K/aggs0.Y,hlth3.pH,hlth3.gradient[0],hlth3.gradient[1],hlth3.gradient[2],hlth3.pTransGood,
       hlth3.pTransBad,eq3.rent,eq3.wage,eq3.tax,aggs3.M*price_eu*aux.copay]
# saving decision rules
size = stats3.ne*stats3.nh*stats3.nk
opt3 = pd.DataFrame(index=np.arange(0,size),columns=['e','h','k','ps','c','m','kp','v'])
opt3.loc[:,'ps'] = eq3.stats.probs
for i,s in enumerate(stats3.states):
    e,h,k = s
    opt3.loc[i,['e','h','k']] = [e,h,k]
    opt3.loc[i,'c']  = eq3.stats.optc[e,h,k]
    opt3.loc[i,'m']  = eq3.stats.optm[e,h,k]
    opt3.loc[i,'kp'] = eq3.stats.optk[e,h,k]
    opt3.loc[i,'v']  = eq3.stats.value[e,h,k]


# In[41]:


eqs3 = []


# In[42]:


scenarios = ['aeu_pe']
results3  = pd.DataFrame(index=outcomes,columns=scenarios)


# In[43]:


#scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
opt3.to_pickle('../output_JPE/opt_aeu_pe3.pkl')
file = open('../output_JPE/eq_aeu_pe3.pkl','wb')
pickle.dump(eq3,file)
file.close()
results3.loc[:,'aeu_pe'] = res3
eqs3.append(eq3)


# In[44]:


results3.loc['r','aeu_pe']


# In[45]:


results3.loc['s','aeu_pe']


# In[46]:


results3.loc['tbad','aeu_pe']


# # Save results

# In[47]:


scenarios = ['pus_ge','peu_ge','peu_pe','aeu_ge','aeu_pe']
ResTot    = pd.DataFrame(index=outcomes,columns=scenarios)


# In[48]:


ResTot.loc[:,'pus_ge'] = results
ResTot.loc[:,'peu_ge'] = results0
ResTot.loc[:,'peu_pe'] = results1
ResTot.loc[:,'aeu_ge'] = results2
ResTot.loc[:,'aeu_pe'] = results3


# In[49]:


ResTot.to_pickle('../output_JPE/welfare_aggregates3.pkl')


# In[ ]:




