import numpy as np
import csv
from scipy.stats import norm
from scipy.optimize import minimize
import itertools
from multiprocessing import Pool as pool
from functools import partial
from scipy.interpolate import interp1d
from model_wage.foncs import utility, invest, closest, evalue
from model_wage import params

class bellman:
    def __init__(self,country='us',flex=None,aux=None,inc=None,
        options=None,taxrate=0.1,wage=1.5,rent=3e-2):
        if flex!=None:
            self.flex = flex
        else :
            self.flex = params.flexpars(country=country)
            #print(vars(self.flex))
        if aux!=None:
            self.aux = aux
        else :
            self.aux = params.auxpars(country=country)
            #print(vars(self.aux))
        if options!=None:
            self.op = options
        else :
            self.op = params.settings()
        if inc!=None:
            self.inc = inc
        else :
            self.inc = params.incprocess(country=country)
            self.inc.tauchen(ne=self.op.ne)
        self.wage = wage
        self.rent = rent
        self.taxrate = taxrate
        self.country = country
        self.set_grids()
        return
    def set_grids(self):
        # some adjustments
        self.flex.delta = [j + self.flex.eta
            * self.aux.risky + self.flex.risk for j in self.flex.delta]
        # grid for kapital
        self.tgridk = np.linspace(0.0,self.op.maxk**self.op.curv,self.op.nk)
        self.gridk = np.array([x**(1.0/self.op.curv) for x in self.tgridk])
        self.gridh = np.arange(0,self.op.nh,1)
        self.gride = np.arange(0,self.op.ne,1)
        grids = [self.gride,self.gridh,np.arange(0,self.op.nk,1)]
        self.states = list(itertools.product(*grids))
        self.compute_cash()
        return
    def set_wage(self,wage):
        self.wage = wage
        self.compute_cash()
        return
    def set_rent(self,rate):
        self.rent = rate
        self.compute_cash()
        return
    def set_tax(self,taxrate):
        self.taxrate = taxrate
        self.compute_cash()
        return
    def compute_cash(self):
        ne = self.op.ne
        nh = self.op.nh
        nk = self.op.nk
        cash = np.zeros((ne,nh,nk))
        for e in range(ne):
            earn = self.wage * np.exp(self.inc.pte[e])
            for h in range(nh):
                cash[e,h,:] = (1.0-self.taxrate)*earn*self.aux.Gamma[h] + self.gridk * (1.0 + self.rent)
        self.cash = cash
        return
    def itervalue(self):
        ne, nh, nk = self.op.ne, self.op.nh, self.op.nk
        shape = (ne,nh,nk)
        self.compute_cash()
        cash = self.cash.copy()
        iter = 1
        while iter<=1:
            if (hasattr(self, 'optc')) :
                cons = self.optc.copy()
            else :
                cons = 0.3*cash
            if (hasattr(self,'optm')):
                medexp = self.optm.copy()
            else :
                medexp = 0.1*cash
            if (hasattr(self, 'value')) :
                value = self.value.copy()
            else :
                value = np.zeros(shape)
            count = 1
            p = pool(self.op.nprocs)
            while count < 250:
                # solve for optimum
                result = p.map(partial(self.getopt2,nextvalue=value,icons=cons,imed=medexp,icash=cash), self.states)
                result = np.asarray(result)
                tcons, tmedexp, tvalue = result[:,0], result[:,1], result[:,2]
                dcons = np.absolute(tcons.reshape(shape)-cons)
                dmedexp = np.absolute(tmedexp.reshape(shape)-medexp)
                if np.max(dcons)<1e-3 and np.max(dmedexp)<1e-3 :
                    cons = tcons.reshape(shape)
                    medexp = tmedexp.reshape(shape)
                    value = tvalue.reshape(shape)
                    break
                else :
                    cons = tcons.reshape(shape)
                    medexp = tmedexp.reshape(shape)
                    value = tvalue.reshape(shape)
                    #fvalue = self.splines(value)
                    #print('count = ',count,np.max(dcons),np.max(dmedexp))
                    count +=1
            p.close()
            if count==250:
                if hasattr(self,'optc'):
                    del self.optc
                    del self.value
                    del self.optm
                iter +=1
            else :
                break
        self.value = value.copy()
        self.optc = cons.copy()
        self.optm = medexp.copy()
        self.optk = np.zeros(shape)
        for s in self.states:
            self.optk[s] = self.cash[s] - self.optc[s] - self.aux.copay*self.flex.price*self.optm[s]
        #print('solved bellman equation (iter, diff cons, diff medexp) ',count,np.max(dcons),np.max(dmedexp))
        #print('grid for k (nk, maxk) ',self.op.nk,np.max(self.gridk))
        return
    def funcvalue(self,flows,state,cash,nextvalue):
        cons, medexp = flows[0], flows[1]
        e, h, k = state
        nextk = cash - cons - self.aux.copay * self.flex.price * medexp
        # probabilities that apply in that state
        klow = np.array(0)
        kup = np.array(0)
        ku = np.array(0.0)
        delta = np.array(self.flex.delta)
        if nextk>=0 and cons>0 and medexp>=0:
            klow,kup,ku = closest(self.op.nk,nextk,self.gridk)
            probh = invest(h,medexp,self.flex.psi,delta)
            value = utility(cons,h,self.flex.sigma,self.flex.phi)
            value += evalue(self.op.ne,self.op.nh,self.op.nk,self.flex.beta,klow,kup,ku,nextvalue,np.array([1.0-probh,probh]),self.inc.tprob[e,:])
        else :
            value = -1e20
        return -value


    def getopt(self, state, nextvalue, icons, imed, icash):
        e, h, k = state
        cash = icash[e,h,k]
        options = {'xtol':1-2}
        #options = None
        # problem 1: unbounded f = 1/(1+exp(-x)), x = -np.log(1/f-1)
        x = [icons[e,h,k],imed[e,h,k]]
        prob1 = minimize(partial(self.funcvalue,state=state,cash=cash,nextvalue=nextvalue),x,method='Nelder-Mead',options=options)
        optc = prob1.x[0]
        optm = prob1.x[1]
        value = prob1.fun
        # problem 2: constrained m = 0
        x = icons[e,h,k]
        prob2 = minimize(partial(self.funcvalue,state=state,cash=cash,nextvalue=nextvalue),x,method='Nelder-Mead',options=options)
        if prob2.fun < value:
            optc = prob2.x[0]
            optm = 0.0
            value = prob2.fun
        return [optc, optm, -value]

    def getopt2(self, state, nextvalue, icons, imed, icash):
        e, h, k = state
        cash = icash[e,h,k]
        minc = 0.01
        maxc = cash
        gridc = np.linspace(minc**self.op.curv,maxc**self.op.curv,20)
        gridc = np.array([c**(1.0/self.op.curv) for c in gridc])
        value = 1e10
        optc = icons[e,h,k]
        optm = imed[e,h,k]
        flex = [self.flex.sigma,self.flex.beta,self.flex.phi,self.flex.psi,self.flex.delta[0],self.flex.delta[1],self.flex.price]
        for c in gridc:
            minm = 0.0
            maxm = (cash - c)/(self.aux.copay*self.flex.price)
            if (maxm<0.0):
                maxm = 0.0
            gridm = np.linspace(minm,maxm**self.op.curv,20)
            gridm = np.array([m**(1.0/self.op.curv) for m in gridm])
            for m in gridm :
                v = self.funcvalue(np.array([c,m]),state,cash,nextvalue)
                #v = -onevalue2(np.array([c,m]),state,cash,nextvalue,self.gridk,self.aux.copay,flex,self.inc.tprob[e,:])
                if (v<value):
                    value = v
                    optc = c
                    optm = m
        x = np.array([optc,optm])
        options = None
        #{'xtol':1-2}
        prob1 = minimize(partial(self.funcvalue,state=state,cash=cash,nextvalue=nextvalue),x,method='Powell',options=options)
        optc = prob1.x[0]
        optm = prob1.x[1]
        value = prob1.fun
        return [optc, optm, -value]
