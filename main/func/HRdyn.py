import numpy as np
from scipy.integrate import odeint

a = 1
b = 3
c = 1
d = 5
s = 4
r = 0.005
p0 = -1.6
I = 3.24
Osyn = 1
lam = 10
Vsyn1 = 2
Vsyn2 = -1.5


def dynamic( y, t, adj, gc ): 
    p, q, n = np.split(y, 3)  
    Tp = 1 / (1 + np.exp(-lam * (p - Osyn)))
    cp = gc * (Vsyn1 - p) * (adj.dot(Tp))  
    dp = q - a * (p ** 3) + b * (p ** 2) - n + I + cp 
    dq = c - d * (p ** 2) - q
    dn = r * (s * (p - p0) - n)
    return np.concatenate([dp, dq, dn])


def time_series( adj, deltaT, gc):
    n = adj.shape[0]
    y0 = np.random.random_sample(3 * n ) 
    t = np.linspace(0, 1200, 1200 * int(1 / deltaT) + 1)
    ts = odeint(dynamic, y0, t, args=(adj, gc))
    ts = ts[-1000 * int(1 / deltaT):]  
    ts = ts.reshape( -1, 3, n ) 
    ts = np.transpose(ts, [2, 0, 1]) 
    return ts[:,:, 0] 

def covariance( time_series ):
    return np.cov( time_series ) 

def func(A, deltaT=0.1, gc=1):
    print( deltaT, gc )
    adj = A
    ts = time_series(  adj, deltaT, gc )
    cov = covariance( ts )
    return cov
