import numpy as np
import csv
from scipy.stats import norm
from scipy.optimize import minimize, brentq, root,fixed_point, minimize_scalar
import itertools
from multiprocessing import Pool as pool
from functools import partial
#from numba import jit
from scipy.interpolate import interp1d
from model_copay.foncs import closest, invest, transition
from model_copay.micro import bellman
from matplotlib import pyplot as plt

class stationary:
    def __init__(self,dp=None,curv=None,nk=100):
        if bellman!=None:
            self.dp = dp
        else :
            self.dp = bellman()
        self.nk = nk
        self.ne = self.dp.op.ne
        self.nh = self.dp.op.nh
        self.maxk = self.dp.op.maxk
        if curv!=None:
            self.curv = curv
        else :
            self.curv = self.dp.op.curv
        self.tgridk = np.linspace(0.0,self.maxk**self.curv,self.nk)
        self.gridk = np.array([k**(1.0/self.curv) for k in self.tgridk])
        self.states = list(itertools.product(*[self.dp.gride,self.dp.gridh,[k for k in range(nk)]]))
        return
    def blowup(self):
        # small grids
        optc = self.dp.optc.copy()
        optm = self.dp.optm.copy()
        optk = self.dp.optk.copy()
        value = self.dp.value.copy()
        # on new grids
        self.optc = np.zeros((self.ne,self.nh, self.nk))
        self.optm = np.zeros((self.ne,self.nh, self.nk))
        self.optk = np.zeros((self.ne,self.nh, self.nk))
        self.value = np.zeros((self.ne,self.nh, self.nk))
        for e in range(self.ne):
            for h in range(self.nh):
                x = self.dp.gridk
                y = optc[e,h,:]
                f = interp1d(x,y,kind='linear')
                self.optc[e,h,:] = f(self.gridk)
                y = optm[e,h,:]
                f = interp1d(x,y,kind='linear')
                self.optm[e,h,:] = f(self.gridk)
                y = optk[e,h,:]
                f = interp1d(x,y,kind='linear')
                self.optk[e,h,:] = f(self.gridk)
                y = value[e,h,:]
                f = interp1d(x,y,kind='linear')
                self.value[e,h,:] = f(self.gridk)
        return
    def compute(self):
        if (hasattr(self,'probs')):
            probs = self.probs.copy()
        else :
            probs = (1/len(self.states))*np.ones(len(self.states))
        count = 1
        ne, nh, nk = self.ne, self.nh, self.nk
        ns = ne * nh * nk
        klow =  np.array(0)
        kup = np.array(0)
        ku = np.array(0.0)
        probh = np.zeros(nh)
        while count < 2000:
            tprobs = np.zeros(len(self.states))
            for i,s in enumerate(self.states):
                e,h,k = s
                probe = self.dp.inc.tprob[e,:]
                probh[1] = invest(h,self.optm[e,h,k],self.dp.flex.psi,np.array(self.dp.flex.delta))
                probh[0] = 1.0-probh[1]
                pi = probs[i]
                klow, kup, ku = closest(nk,self.optk[e,h,k],self.gridk)
                if ku<0.5:
                    kk = klow
                else :
                    kk = kup
                tprobs += transition(kk,ne,nh,nk,ns,probe,probh,pi)
            criterion = np.absolute((tprobs - probs))
            if np.max(criterion)<1e-6:
                probs = tprobs.copy()
                break
            else :
                probs = tprobs.copy()
                count +=1
        #print('stationary distribution converged in ', count, ' iterations ')
        #print('grid for k (nk maxk)', len(self.gridk), np.max(self.gridk))
        self.probs = probs
        #plt.plot(self.gridk,self.probs[0:100])
        #plt.show()
        return

    def getprob(self,state,probs):
        e,h,k = state
        ne, nh, nk = self.ne, self.nh, self.nk
        ns = ne * nh * nk
        probe = self.dp.inc.tprob[e,:]
        probh = np.zeros(nh)
        probh[1] = invest(h,self.optm[e,h,k],self.dp.flex.psi,np.array(self.dp.flex.delta))
        probh[0] = 1-probh[1]
        pi = probs[e*self.nh*self.nk + h*self.nk + k]
        ns = self.nh * self.ne * self.nk
        nextk = self.optk[e,h,k]
        klow = np.array(0)
        kup = np.array(0)
        ku = np.array(0.0)
        klow, kup, ku = closest(nk, nextk, self.gridk)
        if ku<0.5:
            kk = klow
        else :
            kk = kup
        return transition(kk,nk,ns,probe,probh,pi)

    def compute2(self):
        if (hasattr(self,'probs')):
            probs = self.probs.copy()
        else :
            probs = (1/len(self.states))*np.ones(len(self.states))
        count = 1
        p = pool(self.dp.op.nprocs)
        while count < 2000:
            tprobs = np.zeros(len(self.states))
            result = p.map(partial(self.getprob,probs=probs), self.states)
            for r in result:
                tprobs += r
            criterion = np.absolute((tprobs - probs))
            if np.max(criterion)<1e-6:
                probs = tprobs.copy()
                break
            else :
                probs = tprobs.copy()
                count +=1
        p.close()
        #print('stationary distribution converged in ', count, ' iterations ')
        self.probs = probs
        return
    def get_kdist(self):
        pk = np.zeros(self.nk)
        for i,s in enumerate(self.states):
            e,h,k = s
            pk[k] += self.probs[i]
        return pk
