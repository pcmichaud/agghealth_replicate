import numpy as np
from scipy.optimize import brentq, root,fixed_point, minimize_scalar
import itertools
from functools import partial
#from numba import jit
from scipy.interpolate import interp1d
from model.foncs import invest
from model.micro import bellman
from model.distributions import stationary
from model.params import flexpars, incprocess, auxpars, settings
from scipy.optimize import minimize
import csv

class health_check:
    def __init__(self):
        self.pH = 0.0
        self.pTransBad = 0.0
        self.pTransGood = 0.0
        self.gradient = None
        return

class aggregates:
    def __init__(self):
        self.C = 0.0
        self.M = 0.0
        self.K = 0.0
        self.N = 0.0
        self.Y = 0.0
        return

class equilibrium:
  def __init__(self,stats=None,inirent=2e-2,initax=None,taxes=False,rent=False):
      if stats!=None:
          self.stats = stats
          self.aux = self.stats.dp.aux
          self.flex = self.stats.dp.flex
          self.op = self.stats.dp.op
          self.inc = self.stats.dp.inc
      else :
          self.stats = stationary()
     # inittal values
      self.inirent = inirent
      if initax==None:
          with open('../moments/'+self.stats.dp.country+'/mshare.csv', mode='r') as infile:
              reader = csv.reader(infile)
              values = [rows for rows in reader]
              mshare = float(values[0][0])
              self.initax =  (1.0 - self.aux.copay) * mshare / (1.0 - self.aux.alpha)
      else :
          self.initax = initax
      self.solve_tax = taxes
      self.solve_rent = rent
      return
  def market(self,rent,tax):
      nagg = self.labor()
      kd = self.demand(rent,nagg)
      wage = self.get_wage(nagg,kd)
      self.stats.dp.set_wage(wage)
      self.stats.dp.set_rent(rent)
      self.stats.dp.set_tax(tax)
      self.stats.dp.itervalue()
      self.stats.blowup()
      self.stats.compute()
      kdist = self.stats.get_kdist()
      ks = np.sum([k*p for k,p in zip(self.stats.gridk,kdist)])
      return kd,ks
  def demand(self,rent,nagg):
      den = self.aux.alpha*self.flex.tfp*(nagg**(1.0-self.aux.alpha))
      num = rent + self.aux.delta_k
      return (num/den)**(1.0/(self.aux.alpha-1.0))
  def labor(self):
      tn = self.inc.tprob.copy()
      pn = (1.0/self.op.ne)*np.ones((self.op.ne,1))
      count = 1
      while count<50:
          tpn = np.transpose(tn).dot(pn)
          criterion = np.max(np.abs(tpn - tn))
          if criterion<1e-4:
              pn = tpn.copy()
              break
          else :
              pn = tpn.copy()
              count +=1
      nagg = np.sum([p*np.exp(e) for p,e in zip(pn,self.inc.pte)])
      return nagg
  def get_wage(self,nagg,kagg):
      alpha = self.aux.alpha
      tfp = self.flex.tfp
      wage = (1.0-alpha) * tfp * (kagg**alpha) * (nagg**(-alpha))
      return wage
  def get_rent(self,nagg,kagg):
      alpha = self.aux.alpha
      tfp = self.flex.tfp
      delta_k = self.aux.delta_k
      rent = alpha * tfp * (kagg**(alpha-1.0))*(nagg**(1.0-alpha))-delta_k
      return rent
  def excess(self,rent,tax):
      kd, ks = self.market(rent,tax)
      #print('rate = ',rent, 'demand = ',kd,'supply = ',ks,' tax = ',tax)
      return kd - ks
  def get_rent(self,tax):
      rmin = 0.0005
      rmax = 1.0/self.flex.beta - 1.0 - 0.005
      while True:
          try:
            rstar = brentq(self.excess,rmin,rmax,xtol=1e-4,args=(tax))
            break
          except ValueError:
              if rmin<=1e-5:
                  rstar = 0.0
                  ex = self.excess(rstar,tax)
                  break
              else :
                rmin *= 1e-1
                rmax *= 0.5
      return rstar
  def get_tax(self,tax):
      rstar = self.get_rent(tax)
      # revenue
      taxbase = self.stats.dp.wage * self.labor()
      # spending
      mtot = 0.0
      for i,s in enumerate(self.stats.states):
          e,h,k = s
          #taxbase += rstar*self.csumers.gridk[k]*self.csumers.probs[i]
          mtot += self.stats.probs[i]*self.stats.optm[e,h,k]
      spe = (1.0-self.aux.copay)*self.flex.price * mtot
      self.rent = rstar
      self.wage = self.stats.dp.wage
      return spe/taxbase
  def solve(self):
      if (self.solve_tax):
          count = 1
          tax = fixed_point(self.get_tax,self.initax,xtol=1e-2)
          self.tax = tax
          self.initax = tax
      else :
          self.tax = self.initax
          if (self.solve_rent):
               self.rent = self.get_rent(self.tax)
               self.wage = self.stats.dp.wage
          else :
               self.rent = self.inirent
               self.wage = self.stats.dp.wage
               kd, ks = self.market(self.rent,self.tax)
      return
  def aggregates(self):
      aggs = aggregates()
      for i,s in enumerate(self.stats.states):
          e,h,k = s
          aggs.C += self.stats.probs[i]*self.stats.optc[e,h,k]
          aggs.M += self.stats.probs[i]*self.stats.optm[e,h,k]
          aggs.K += self.stats.probs[i]*self.stats.gridk[k]
      aggs.N = self.labor()
      aggs.Y = self.flex.tfp * (aggs.K**self.aux.alpha) * (aggs.N**(1.0-self.aux.alpha))
      return aggs
  def healthreport(self):
      doctor = health_check()
      for i,s in enumerate(self.stats.states):
          e,h,k = s
          trans = invest(h,self.stats.optm[e,h,k],self.flex.psi,np.array(self.flex.delta))
          if h==1:
              doctor.pH += self.stats.probs[i]
              doctor.pTransGood += self.stats.probs[i]*trans
          else :
              doctor.pTransBad += self.stats.probs[i]*trans
      doctor.pTransGood = doctor.pTransGood/doctor.pH
      doctor.pTransBad = doctor.pTransBad/(1.0-doctor.pH)

      # fraction in good health by earnings
      ne = self.stats.ne
      phe = np.zeros(ne)
      pe = np.zeros(ne)
      for i,s in enumerate(self.stats.states):
          e, h, k = s
          if h==1:
              phe[e] += self.stats.probs[i]
          pe[e] += self.stats.probs[i]
      phe  = [phe[e]/pe[e] for e in range(ne)]
      cdfe = np.zeros(ne)
      cdfe[0] = pe[0]
      for e in range(1,ne):
          cdfe[e] = cdfe[e-1] + pe[e]
      g = interp1d(cdfe,phe,kind='linear')
      quintiles = [0.125,0.375,0.625,0.875]
      gradients = np.zeros(4)
      for i,q in enumerate(quintiles):
        gradients[i] = g(q)
      for i in range(1,4):
          gradients[i] = gradients[i]/gradients[0]
      doctor.gradient = gradients[1:]
      return doctor
