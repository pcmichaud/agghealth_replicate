
import numpy as np
import csv
from scipy.stats import norm
from collections import OrderedDict
import pandas as pd 
import os
module_dir = os.path.dirname(os.path.dirname(__file__))

class flexpars:
    def __init__(self,country='us',sigma=None,beta=None,phi=None,psi=None,
        delta_h1=None,delta_h2=None,eta=None,tfp=None,price=None,risk=None):
        self.country = country
        self.sigma = 2.1
        self.beta = 0.97
        self.phi = 0.397
        self.psi = 0.161
        self.delta = [-1.0,3.51]
        self.tfp = 1.0
        self.eta = 0.0
        self.price = 1.0
        self.risk = 0.0
        if sigma!=None:
            self.sigma = sigma
        if beta!=None:
            self.beta = beta
        if phi!=None:
            self.phi = phi
        if psi!=None:
            self.psi = psi
        if delta_h1!=None:
            self.delta[0] = delta_h1
        if delta_h2!=None:
            self.delta[1] = delta_h2
        if eta!=None:
            self.eta = eta
        if tfp!=None:
            self.tfp = tfp
        if price!=None:
            self.price = price
        if risk!=None:
            self.risk = risk
        return
class incprocess:
    def __init__(self,country='us',mean=None,rho=None,sige=None):
        self.country = country
        pars = pd.read_pickle(module_dir+'/model/params/income_shocks.pkl')
        pars = pars[self.country.upper()]
        self.rho = pars['rho']
        self.sige = pars['sige']
        if mean!=None:
            self.mean = mean
        else :
            self.mean = 0.0
        if rho!=None:
            self.rho = rho
        if sige!=None:
            self.sige = sige
        self.sige = np.sqrt(self.sige)
        self.sigs = np.sqrt((self.sige**2)/(1.0-self.rho**2))
        return
    def tauchen(self,ne=10,m=2.5):
        sige = self.sige
        rho = self.rho
        mu = self.mean
        Phi = norm.cdf
        if ne==1:
            self.pte = self.mean
            self.tprob = 1.0
        else :
            ptemax = m * np.sqrt((sige**2)/(1.0-rho**2))
            ptemin = -ptemax
            step = (ptemax - ptemin)/(ne-1)
            pte = np.linspace(ptemin,ptemax,ne)
            pte = pte + mu/(1.0-rho)
            tprob = np.zeros((ne,ne))
            for j in range(ne):
                for k in range(ne):
                    if k==0:
                        eps = pte[0] - mu - rho*pte[j] + 0.5*step
                        tprob[j][k] = Phi(eps/np.sqrt(sige**2))
                    elif (k==ne-1):
                        eps = pte[ne-1] - mu - rho*pte[j] - 0.5*step
                        tprob[j][k] = 1.0 - Phi(eps/np.sqrt(sige**2))
                    else :
                        eps = pte[k] - mu - rho*pte[j]
                        tprob[j][k] = Phi((eps+0.5*step)/np.sqrt(sige**2))-Phi((eps-0.5*step)/np.sqrt(sige**2))
            self.pte = pte.copy()
            self.tprob = tprob.copy()
        cprob = np.zeros((ne,ne))
        for e in range(ne):
            cprob[e,0] = tprob[e,0]
            for j in range(1,ne):
                cprob[e,j] = cprob[e,j-1] + tprob[e,j]
        self.cprob = cprob.copy()
        return
class auxpars:
    def __init__(self,country='us',copay=None,delta_k=None,alpha=None,risky=None,copay_max=1.2,copay_min=0.8):
        self.country = country
        pars = pd.read_pickle(module_dir+'/model/params/auxiliary.pkl')
        pars = pars[self.country.upper()]
        self.copay = pars['mu']
        self.copay_min = copay_min * self.copay
        self.copay_max = copay_max * self.copay
        self.copays = np.linspace(self.copay_min,self.copay_max,10)
        self.delta_k = pars['delta']
        self.alpha = pars['alpha']
        self.risky = 0.0
        if copay!=None:
            self.copay = copay
        if delta_k!=None:
            self.delta_k = delta_k
        if alpha!=None:
            self.alpha = alpha
        if risky!=None:
            self.risky = risky
        return
class settings:
    def __init__(self,ne=10,nk=30,nh=2,curv=0.5,maxk=100.0,nprocs=40):
        self.ne = ne
        self.nk = nk
        self.nh = nh
        self.curv = curv
        self.maxk = maxk
        self.nprocs =nprocs
        return
