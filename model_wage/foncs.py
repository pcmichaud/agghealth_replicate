import numpy as np 
from numba import njit, jit

@njit
def utility(cons, h, sigma, phi):
    if sigma!=1.0:
        if cons > 0.0:
            uf = (cons**(1.0-sigma))/(1.0-sigma)
        else :
            uf = -1e10
    else :
        if cons > 0.0:
            uf = np.log(cons)
        else :
            uf = -1e10       
    if h==1:
        uf += phi 
    return uf

@njit
def invest(h,medexp,psi,delta):
    hstar = psi * medexp + delta[h]
    if hstar < 0.0:
        hstar = 0.0
    inv = 1.0 - np.exp(-hstar)
    return inv

@jit 
def closest(nx,x,grid):
    ilow = nx-1 
    iup = nx-1 
    xu = 0.0 
    for i in range(nx):
        if x <= grid[i]:
            ilow = i-1
            if ilow<0: 
                ilow = 0
            iup = i 
            if ilow!=iup:
                xu = (x-grid[ilow])/(grid[iup]- grid[ilow])
            else :
                xu = 0.0 
            break
    return ilow,iup,xu

@njit 
def evalue(ne,nh,nk,beta,klow,kup,ku,nextvalue,probh,probe):
    value = 0.0 
    for e in range(ne):
        for h in range(nh):
            value += beta*((1.0-ku)*nextvalue[e,h,klow]+ ku*nextvalue[e,h,kup])*probh[h]*probe[e]
    return value 



@njit
def transition(kk,ne,nh,nk,ns,probe,probh,probs):
    tprobs = np.zeros(ns)
    for ee in range(ne):
        for hh in range(nh):
            j = ee*nh*nk + hh*nk + kk
            tprobs[j] += probe[ee]*probh[hh]*probs
    return tprobs
