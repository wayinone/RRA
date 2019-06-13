import numpy as np
from numpy import array
import scipy.odr as odr
import matplotlib.pyplot as plt
import time
import sympy as sp
from scipy import stats

def line(t,b0,b1,b2,b3):
    x = b0 + b2*t
    y = b1 + b3*t
    return x,y
def line_ini(x,y):
    d=2;K=1
    X_nd = np.hstack((x[:,None],y[:,None]))
    bparm = np.zeros(d+K*d)
    Xbari = np.mean(X_nd,axis=0)#1-by-d
    bparm[:d]=Xbari
    Bi = X_nd-Xbari
    BBi = np.dot(Bi.T,Bi)
    D,V = np.linalg.eig(BBi)
    ind = np.argsort(D)#sort from small to big
    for k in np.arange(K):
        v = V[:,ind[-k-1]]# the biggest eigenvector
        v = v[...,None]
        bparm[d+k*d:d+(k+1)*d]=v.flatten()*np.sign(v[0])
    return bparm

def quadratic(t,b0,b1,b2,b3):
    """
    len_b = 4
    
    input: 
        t: 
        b 
            b0,b1: origin
            b2: parameter for quodratic term, b[2]>0
            b3: rotation angle (e.g. pi/2, pi,...)
    """
    xx = t
    yy = b2*t**2
    
    s = sp.sin(b3)
    c = sp.cos(b3)
    x = xx*c-yy*s+b0
    y = xx*s+yy*c+b1
    return x,y
   
def quadratic_ini(x,y,w=None):
    if w is None:
        sw=1
    else:
        sw = np.sqrt(w)
    x = x.astype('float64')
    y = y.astype('float64')
    n_rot = 32
    rot = np.array([2*i*np.pi/n_rot for i in range(n_rot)]) 
    X_dn = np.vstack((x,y))
    n = x.shape[0]
    res = np.zeros((n_rot,))
    coef= np.zeros((n_rot,3))
    for i in range(n_rot):
        Xr_dn = rotation(X_dn,rot[i])
        xx = Xr_dn[0,:]
        yy = sw*Xr_dn[1,:]
        A = sw*np.array([np.ones((n,)),xx,xx**2])
        beta, residual,rank,s = np.linalg.lstsq(A.T,yy)
        coef[i,:]=beta
        try:
            tmp = residual[0]
        except:
            tmp=np.max(res)
        res[i] = tmp
    
    ind = np.argmin(res)
    c = coef[ind,:]
    if c[2]<1e-2:
        c[2]=1e-2
    b0r = -c[1]/2/c[2]
    b1r = c[0]-c[1]**2/4/c[2]
    
    b2 =c[2]
    b3 = -rot[ind]   
    center=rotation(np.array([[b0r],[b1r]]),b3).flatten()
    b0=center[0]
    b1=center[1]
    B0 = np.array([b0,b1,b2,b3])
    return B0

def quadratic_stdfy(bparm):
    # bparm is of size(n,4)
    bstdfy = bparm.copy()
    b2 = bparm[:,2]
    b3 = bparm[:,3]
    bstdfy[:,2]=np.abs(b2)
    bstdfy[:,3]=np.mod(b3+np.pi*(b2<0),2*np.pi)
    return bstdfy

def quadratic3D(t,x0,y0,z0,a,b,c,beta):
    """
    len_b = 7
    """
    xx = t
#    yy = 0
    zz = beta*t**2
    sa = sp.sin(a)
    ca = sp.cos(a)
    sb = sp.sin(b)
    cb = sp.cos(b)
    sc = sp.sin(c)
    cc = sp.cos(c)
    x = cb*cc*xx  + (sa*sc-ca*sb*cc)*zz+x0
    y = -cb*sc*xx + (sa*cc+ca*sb*sc)*zz+y0
    z = sb*xx+ca*cb*zz+z0
    return x,y,z

def quadratic3D_ini(x,y,z,w=None):
    if w is None:
        sw=1
    else:
        sw = np.sqrt(w)
    x = x.astype('float64')
    y = y.astype('float64')
    n_th = 4
    rot = np.array([i*np.pi/n_th for i in range(n_th)]) 
    n_rot = n_th**3
    X_dn = np.vstack((x,y,z))
    n = x.shape[0]
    res = np.zeros((n_rot,))
    coef= np.zeros((n_rot,3))
    rotijk = np.zeros((n_rot,3))
    count=0
    for i in range(n_th):
        for j in range(n_th):
            for k in range(n_th):
                Xr_dn = rotation_3D(X_dn,rot[i],rot[j],rot[k])
                xx = Xr_dn[0,:]
                zz = sw*Xr_dn[1,:]
                A = sw*np.array([np.ones((n,)),xx,xx**2])
                beta, residual,rank,s = np.linalg.lstsq(A.T,zz)
                rotijk[count,:]=np.array([rot[i],rot[j],rot[k]])
                coef[count,:]=beta
                res[count] =  residual[0]
                count +=1
    
    ind = np.argmin(res)
    c = coef[ind,:]
    x0r = -c[1]/2/c[2]
    z0r = c[0]-c[1]**2/4/c[2]
    
    beta =c[2]
    a = -rotijk[ind,0]
    b = -rotijk[ind,1]
    c = -rotijk[ind,2]
    center=rotation_3D(np.array([[x0r],[0],[z0r]]),a,b,c).flatten()
    x0=center[0]
    y0=center[1]
    z0=center[2]
    B0 = np.array([x0,y0,z0,a,b,c,beta])
    return B0

def rotation(X,th):
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    Xr = np.dot(R,X)
    return Xr
def rotation_3D(X,a,b,c):
    sa = np.sin(a)
    ca = np.cos(a)
    sb = np.sin(b)
    cb = np.cos(b)
    sc = np.sin(c)
    cc = np.cos(c)
    R = np.array([[cb*cc,ca*sc+sa*sb*cc,sa*sc-ca*sb*cc],[-cb*sc,ca*cc-sa*sb*sc,sa*cc+ca*sb*sc],[sb,-sa*cb,ca*cb]])
    Xr = np.dot(R,X)
    return Xr