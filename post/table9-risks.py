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

countries = ['us','fr','sp','it','dk','de','nl','se']
results = pd.DataFrame(index=['s','h','g','coy'],columns=countries)

df = pd.read_pickle('../estimation/output/params_ref_us.pkl')
pars = df.loc[:,'value'].to_frame()
pars.columns = ['us']
for c in countries:
    df = pd.read_pickle('../estimation/output/params_ref_'+c+'.pkl')
    pars[c] = df.loc[:,'value']
print(pars)


def eq(co,iprice=None,itfp=None,irisks=None):
    # estimated parameters
    p = pars.loc[:,co]
    if iprice!=None or itfp!=None or irisks!=None:
        itax = True
    else :
        itax = False
    if iprice==None:
        iprice = p['price']
    if itfp==None:
        itfp = p['tfp']
    if irisks==None:
        irisks = [p['delta_h1'],p['delta_h2']]
    theta = params.flexpars(sigma=p['sigma'],beta=p['beta'],
                        phi=p['phi'],psi=p['psi'],delta_h1=irisks[0],
                        delta_h2=irisks[1],eta=0.0,tfp=itfp,price=iprice)
    # option for the numerical solution
    ne = 10
    m  = 2.5
    if co == 'us':
        op = params.settings(ne=ne,nk=30,maxk=190.0,curv=0.5,nprocs=64)
    else:
        op = params.settings(ne=ne,nk=30,maxk=150.0,curv=0.5,nprocs=64)
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
    s    = iprice*aggs.M/aggs.Y
    coy  = (aggs.C+iprice*aggs.M)/aggs.Y
    h    = hlth.pH
    g    = hlth.gradient[2]
    return s,h,coy,g

eu_countries = [c for c in countries if c!='us']

d1_eu = pars.loc['delta_h1',eu_countries].mean()
d2_eu = pars.loc['delta_h2',eu_countries].mean()
print('deltas = ',[d1_eu,d2_eu])

for i,co in enumerate(countries):
    s,h,coy,g = eq(co,irisks=[d1_eu,d2_eu])
    print('country = ',co,', s = ',s,', h = ',h,', grad4 = ',g)
    results.loc['s',co] = s
    results.loc['h',co] = h
    results.loc['coy',co] = coy
    results.loc['g',co] = g

print(results)

print('saving results...')
results.to_pickle('output/table9-risks.pkl')

