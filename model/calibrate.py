from model import micro
from model import macro
from model import params
from model import distributions as dist
import numpy as np
from scipy.optimize import minimize
import csv
from copy import deepcopy
import nlopt  as nl
import os
module_dir = os.path.dirname(os.path.dirname(__file__))

class point:
    def __init__(self):
        self.name = None
        self.value = None
        self.ifree = None
        self.low = None
        self.up = None

class moment:
    def __init__(self):
        self.name = None
        self.data = None
        self.se = None
        self.sim = None

class initpars:
    def __init__(self,flex):
        self.flex = flex
        self.names = ['sigma','beta','phi','psi','delta_h1',
            'delta_h2','eta','tfp','price','risk']
        self.pars = []
        for p in self.names:
            this = point()
            this.name = p
            if this.name=='delta_h1':
                this.value = self.flex.delta[0]
            elif this.name=='delta_h2':
                this.value = self.flex.delta[1]
            else :
                this.value = getattr(self.flex,p)
            this.ifree = False
            this.low = this.value*0.9
            this.up = this.value*1.1
            self.pars.append(this)
        return
    def fix(self,name,value=None):
        for p in self.pars:
            if p.name == name:
                if value!=None:
                    p.value = value
                p.ifree = False
        return
    def free(self,name,value=None):
        for p in self.pars:
            if p.name == name:
                if value!=None:
                    p.value = value
                p.ifree = True
        return
    def extract_theta(self):
        theta = []
        for p in self.pars:
            if p.ifree:
                theta.append(p.value)
        return theta
    def put_theta(self,theta):
        j = 0
        for p in self.pars:
            if p.ifree:
                p.value = theta[j]
                j+=1
        return
    def extract_low(self):
        theta = []
        for p in self.pars:
            if p.ifree:
                theta.append(p.low)
        return theta
    def extract_up(self):
        theta = []
        for p in self.pars:
            if p.ifree:
                theta.append(p.up)
        return theta
    def set_flex(self):
        for p in self.pars:
            if p.name=='delta_h1':
                self.flex.delta[0] = p.value
            elif p.name=='delta_h2':
                self.flex.delta[1] = p.value
            else :
                setattr(self.flex,p.name,p.value)
        return
    def print(self):
        print('current parameter status : ')
        for p in self.pars:
            print(p.name,np.round(p.value,4),p.ifree)
        return


class msm:
    def __init__(self,country='us',initpar=None,verbose=False,nprocs=40,ge=True):
        self.country = country
        self.initpar = []
        self.verbose = verbose
        self.ge = ge
        # need load initial parameters
        if verbose: print('* Info on parameters initialized')
        if initpar!=None:
            self.initpar = initpar
            self.flex = self.initpar.flex
        else :
            self.flex = params.flexpars(country=country)
            self.initpar = initpars(self.flex)
        self.npar = len(self.initpar.pars)
        self.nfreepar = [p.ifree for p in self.initpar.pars].count(True)
        self.parnames = [p.name for p in self.initpar.pars]
        print('-- ')
        print('number of parameters: ',self.npar)
        print('number of free parameters: ',self.nfreepar)
        self.initpar.print()
        if self.country=='us':
            op = params.settings(nprocs=nprocs,nk=30,curv=0.5,maxk=190.0)
        else:
            op = params.settings(nprocs=nprocs,nk=30,curv=0.5,maxk=150.0)
        self.op = op
        inc = params.incprocess(country=self.country)
        inc.tauchen(ne=op.ne,m=2.5)
        self.inc = inc
        aux = params.auxpars(country=self.country)
        self.aux = aux
        return
    def set_moments(self,moms):
        self.moments = []
        self.nmoms = len(moms)
        print('- using these moments:')
        for n in moms.index:
            this = moment()
            this.name = n
            this.data = moms.loc[n,'mean']
            this.se = moms.loc[n,'sd']
            print(n,this.data,this.se)
            self.moments.append(this)
        return
    def criterion(self,theta,grad):
        # get solution
        self.initpar.put_theta(theta)
        self.initpar.set_flex()
        csumers = micro.bellman(options=self.op,flex=self.initpar.flex,inc=self.inc,aux=self.aux,rent=3e-2)
        stats = dist.stationary(dp=csumers,nk=100)
        self.eq = macro.equilibrium(stats=stats,initax=self.initax,inirent=1.5e-2,rent=self.ge,taxes=False)
        self.eq.solve()
        aggs = self.eq.aggregates()
        report = self.eq.healthreport()
        if self.country=='us':
            f = open(module_dir+'/model/params/sim_gdp_us.csv','w')
            f.write('{}'.format(aggs.Y))
            f.close()
        # building simulated moments
        distance = 0.0
        for m in self.moments:
            if m.name=='cshare':
                m.sim = (aggs.C + aggs.M*self.flex.price)/aggs.Y
            if m.name=='mshare':
                m.sim = aggs.M/aggs.Y*self.flex.price
            if m.name=='kshare':
                m.sim = aggs.K/aggs.Y
            if m.name=='trans_frombad':
                m.sim = report.pTransBad
            if m.name=='trans_fromgood':
                m.sim = report.pTransGood
            if m.name=='grad2':
                m.sim = report.gradient[0]
            if m.name=='grad3':
                m.sim = report.gradient[1]
            if m.name=='grad4':
                m.sim = report.gradient[2]
            if m.name=='tfp':
                if self.country=='us':
                    m.sim = 1.0
                else :
                    f = open(module_dir+'/model/params/sim_gdp_us.csv','r')
                    tfp_us = float(f.readline())
                    f.close()
                    m.sim = aggs.Y/tfp_us
            distance += ((m.data - m.sim)/m.se)**2
        print('f = ',distance,', pars = ',theta)
        print('- current state of moments (data, sim):')
        for m in self.moments:
            print(m.name,m.data,m.sim)
        del self.eq.stats.dp.optc
        del self.eq.stats.dp.optm
        del self.eq.stats.dp.value
        return distance

    def criterion_moms(self,theta):
        # get solution
        self.initpar.put_theta(theta)
        self.initpar.set_flex()
        self.eq.stats.flex = self.flex
        self.eq.stats.dp.flex = self.flex
        self.eq.solve()
        aggs = self.eq.aggregates()
        report = self.eq.healthreport()
        # building simulated moments
        moms = []
        for m in self.moments:
            if m.name=='cshare':
                m.sim = (aggs.C+self.flex.price*aggs.M)/aggs.Y
            if m.name=='mshare':
                m.sim = aggs.M/aggs.Y*self.flex.price
            if m.name=='kshare':
                m.sim = aggs.K/aggs.Y
            if m.name=='trans_frombad':
                m.sim = report.pTransBad
            if m.name=='trans_fromgood':
                m.sim = report.pTransGood
            if m.name=='grad2':
                m.sim = report.gradient[0]
            if m.name=='grad3':
                m.sim = report.gradient[1]
            if m.name=='grad4':
                m.sim = report.gradient[2]
            if m.name=='tfp':
                if self.country!='us':
                    f = open(module_dir+'/model/params/sim_gdp_us.csv','r')
                    tfp_us = float(f.readline())
                    f.close()
                    m.sim = aggs.Y/tfp_us
            moms.append(m.data - m.sim)
        del self.eq.stats.dp.optc
        del self.eq.stats.dp.optm
        del self.eq.stats.dp.value
        return np.array(moms)

    def estimate(self,maxeval=10000):
        for m in self.moments:
            if m.name == 'mshare':
                mshare = m.data
        initax =  (1.0 - params.auxpars(country=self.country).copay) * mshare / (1.0 - params.auxpars(country=self.country).alpha)
        self.initax = initax
        theta = self.initpar.extract_theta()
        low = self.initpar.extract_low()
        up = self.initpar.extract_up()
        n = self.nfreepar
        simp = np.zeros((n+1,n))
        dx = np.zeros(n)
        simp[0,:] = theta
        eps = 0.1
        j = 1
        for i,p in enumerate(self.initpar.pars):
            if p.ifree:
                if self.country=='us':
                    simp[j,:] = theta
                    if p.name=='sigma':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                    if p.name=='beta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='phi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='psi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='delta_h1':
                        dx[j-1] = 0.01#0.05
                        simp[j,j-1] += 0.05
                    if p.name=='delta_h2':
                        dx[j-1] = 0.01#0.1
                        simp[j,j-1] += 0.1
                    if p.name=='eta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='tfp':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.01
                    if p.name=='price':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                    if p.name=='risk':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                if self.country=='nl':
                    simp[j,:] = theta
                    if p.name=='sigma':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                    if p.name=='beta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='phi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='psi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='delta_h1':
                        dx[j-1] = 0.1#0.05
                        simp[j,j-1] += 0.05
                    if p.name=='delta_h2':
                        dx[j-1] = 0.1#0.1
                        simp[j,j-1] += 0.1
                    if p.name=='eta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='tfp':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.01
                    if p.name=='price':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.25
                    if p.name=='risk':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                else:
                    simp[j,:] = theta
                    if p.name=='sigma':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                    if p.name=='beta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='phi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='psi':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.1
                    if p.name=='delta_h1':
                        dx[j-1] = 0.5
                        simp[j,j-1] += 0.05
                    if p.name=='delta_h2':
                        dx[j-1] = 0.5
                        simp[j,j-1] += 0.1
                    if p.name=='eta':
                        dx[j-1] = 0.01
                        simp[j,j-1] += 0.01
                    if p.name=='tfp':
                        dx[j-1] = 0.1
                        simp[j,j-1] += 0.01
                    if p.name=='price':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                    if p.name=='risk':
                        dx[j-1] = 0.25
                        simp[j,j-1] += 0.25
                j +=1

        opt = nl.opt('LN_NEWUOA',n)
        opt.set_min_objective(self.criterion)
        opt.set_initial_step(dx)

        opt.set_maxeval(maxeval)
        opt.set_xtol_abs(1e-4)
        xopt = opt.optimize(theta)
        if opt.last_optimize_result()>0:
            self.opt_theta = xopt
            self.opt_distance = opt.last_optimum_value()
        else :
            self.opt_theta = theta
            self.opt_ditance = np.nan
            print('estimation did not converge, returns flag ',opt.last_optimize_result())

        #opt = minimize(self.criterion,theta,method='BFGS',options={'gtol':1.0} )
        #options={'initial_simplex': simp})

        self.initpar.put_theta(self.opt_theta)
        self.initpar.set_flex()
        return

    def covar(self):
        for m in self.moments:
            if m.name == 'mshare':
                mshare = m.data
        initax =  (1.0 - params.auxpars(country=self.country).copay) *mshare / (1.0 -params.auxpars(country=self.country).alpha)
        self.eq.initax = initax
        thetas = self.initpar.extract_theta()
        n = self.nfreepar
        eps = 1e-3*np.ones(n)
        if self.country=='sp':
            eps = 1e-3*np.ones(n)
        G = np.zeros((self.nmoms,n))
        mbase = self.criterion_moms(thetas)
        # compute G (matrix of derivatives)
        for k in range(n):
            thetas_up = thetas[:]
            if self.country=='sp':
                step = eps[k]
                thetas_up[k] = thetas_up[k] + step
            else:
                step = eps[k]
                thetas_up[k] = thetas_up[k]+eps[k]
            mup = self.criterion_moms(thetas_up)
            G[:,k] = (mup - mbase)/step
        # compute weight matrix
        W = np.zeros((self.nmoms,self.nmoms))
        for i,m in enumerate(self.moments):
            W[i,i] = 1/(m.se**2)
        # compute covar
        Cov = np.linalg.inv(G.transpose() @ W @ G)
        se = np.sqrt(np.diag(Cov))
        self.se = se
        return

