# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:23:44 2016

@author: wayin
"""

import numpy as np
import scipy.odr as odr
import time
import sympy as sp
import scipy.optimize
import inspect
from numpy import array,sin,cos,vstack,ones,zeros
import dill
import matplotlib.pyplot as plt

class EM_cluster(object):
    """
    Input:
        data: the class accept 2 kind of objects:
            1. tuple (x0,x1,...,xd), each xi is of size (n,) for i=1,...,d
            2. (n,d) numpy array
        f   : Only accecpt function f:R1-->Rd
        bini_fcn: function from data (x0,x1,...,xd) t0 the initial b for leastsq
        """
    def __init__(self,data,parms,f,bini_fcn,randomseed=1):
        """
        data: (n,d) matrix
        parms = ['n_clusters','beta_ini','sigma2_ini','lam_ini']
            parms['n_clusters']: number of clusters, say K.
            parms['beta_ini']: size(K,m), m is length of parameter in f. (Optional)
            parms['sigma2_ini]: size (K,). (Optional)
            parms['lam_ini']: size(K,). (Optional)
        """
        np.random.seed(randomseed)
        if type(data)==tuple:
            X_nd = np.vstack(data).T.astype('float64')
        else:
            X_nd = data
        n,d = X_nd.shape        
        self.d = d
        self.n = n
        self.X_nd = X_nd
        self.data = data
        self.parms = parms
        self.f = f        
        self.bini_fcn = bini_fcn
        
        
        K = parms['n_clusters']
        self.K=K
        if  'lam_ini' in parms:
            self.lam_ini=parms['lam_ini']
        else:
#            lam_ini=np.random.rand(K,)
#            lam_ini = lam_ini/np.sum(lam_ini)
            lam_ini = np.ones((K,))/K            
            self.lam_ini = lam_ini
            
        f_inputArg=inspect.getargspec(f)
        self.m = len(f_inputArg[0])-1
        #m: length of paramters used in f
        self.Eq_2find_t()
        
        if 'beta_ini' in parms:
            self.beta_ini= parms['beta_ini']
        else:
            beta_Km = zeros((K,self.m))
            randnum = np.random.rand(n,)
            edge = np.append(0,np.cumsum(lam_ini))
            for k in range(K):
                tmp = (edge[k]<randnum)*(randnum<edge[k+1])
                idk = np.nonzero(tmp)
                Xk = X_nd[idk[0],:]
                beta_Km[k,:]=bini_fcn(*list(Xk.T))
            self.beta_ini = beta_Km
        if 'sigma2_ini' in parms:
            self.sigma2_ini = parms['sigma2_ini']
        else:
            self.sigma2_ini = ones((K,))
                
        self.model_name = f.__name__
        
    def Eq_2find_t(self):
        t = sp.Symbol('t')
        b = sp.symbols('b0:%d' % self.m)
        fmat = sp.Matrix(self.f(t,*b))#size (d,1)
        dfmat = sp.diff(fmat,t)#size (d,1)
        
        self.dim = len(fmat)
        x = sp.symbols('x0:%d' % self.d)
        pt = sp.Matrix(x)#(d,1)
        gg  = (pt-fmat).T*dfmat
        self.sol = sp.Matrix(sp.solve(gg[0],t))# too slow
        self.nsol = len(self.sol)
        self.sollam = sp.lambdify(b+x,self.sol,'numpy')
        self.g  = sp.lambdify(b+x,gg[0],'sympy')
        dg = sp.diff(gg[0],t)
        self.dg = sp.lambdify(b+x,dg,'sympy')
        tb = tuple([t])+b
        self.fnp = sp.lambdify(tb,fmat,'numpy')
        
    def choose_tmat(self,X_nd,ttmat,listb):
        """
        ttmat: size(nsol,n)
        """
        X_nd = X_nd.real
        n = X_nd.shape[0]
        Xpred = self.fnp(ttmat,*listb)# Xpred is array of size(dim,1,nsol,n)
        Xpred = np.squeeze(Xpred,axis=1)# now Xpred is array of size(dim,nsol,n)
        err = np.zeros((self.nsol,n,self.d))
        for d in range(self.d):
            err[:,:,d] = Xpred[d,:,:]-X_nd[:,d]# of size (nsol,n,dim)
        err2 = np.sum(err**2,axis=2)# size  (nsol,n)
        tind = np.argmin(err2,axis=0) #size (n,)
        tmp = range(n)
        tfit = ttmat[tind,tmp]
        pterr = err[tind,tmp,:]# size(n,dim)
        return tfit,pterr
        
    def find_t(self,X_nd,listb):
        """
        f(t,b): t-->(x1(t),x2(t),...,xd(t)), the return should be d-dim tuple
        pt: size(d,)
        b: list of variables of f
        data: tuple (x,y), x and y are of size (n,)
        """
        X_nd = X_nd.astype('complex')
        listx = list(X_nd.T)
        ttmat = self.sollam(*(listb+listx)).real
        ttmat = np.squeeze(ttmat,axis=1)#size(self.nsol,n)
        t_fit,pterr = self.choose_tmat(X_nd,ttmat,listb)
        return t_fit,pterr
    
    def find_t_lite(self,pt,listb):
        """
        lite version of find_t, where pt is (d,) array
        """
        
        pt = pt.astype('complex')
        args=listb+list(pt.flatten())
        tti = self.sollam(*args)
        ttmat = tti.real.flatten() 
        
        Xpred = self.fnp(ttmat,*listb)# Xpred is array of size(dim,1,nsol)
        Xpred = np.squeeze(Xpred,axis=1)# now Xpred is array of size(dim,nsol)
        err = Xpred.T-pt.real
        err2 = np.sum(err**2,axis=1)
        tind = np.argmin(err2)
        tfit = tti[tind]
        return tfit.real,np.min(err2)
        
        
    def pdf_z_given_beta(self,beta_Km,sigma2_K):
        X_nd = self.X_nd
        K = self.K
        n = self.n
        pdf_Kn = zeros((K,n))
        s2 = sigma2_K
        for k in range(K):
            listb = list(beta_Km[k,:])
            _,pterr = self.find_t(X_nd,listb)# size(n,d)
            err2 = np.sum(pterr**2,axis=1)
            pdf_Kn[k,:]=np.exp(-err2**2/2/s2[k])/np.sqrt(2*np.pi*s2[k])
        return pdf_Kn
    
    def err_fun(self,b,X_nd,w=None):
        """
        X_nd has to be a complex matrix of size (n,d)
        """
        listb = list(b)
        _,pterr_nd = self.find_t(X_nd,listb)
        err_flat = pterr_nd.flatten('F')
        if w is not None:
            err_flat = w*err_flat
        return err_flat.astype('float64')
        
    def fit_b(self,x,w=None,initial=None):
        if initial != None:
            B0 = initial
        else:
            xarg = list(x.T)
            n = x.shape[0]
            if w is not None:
#                w1 = w[:n]**2 #in fcn_fit, we will have weight be square rooted
                w1 = w[:n]
                xarg.append(w1)
            B0 = self.bini_fcn(*xarg)
            
        Bfit,_ = scipy.optimize.leastsq(self.err_fun, B0, args=(x.astype('complex'),w),Dfun=None,maxfev=300,)
        
        return Bfit  
    def M_step(self,w_Kn):
        X_nd = self.X_nd
        K = self.K
        m = self.m
        n = self.n
        beta_Km = zeros((K,m))
        sigma2_K=zeros((K,))
        ww = (w_Kn.T/np.sum(w_Kn,axis=1)).T#(K,n)
        for k in range(K):
            wei = np.append(ww[k,:],ww[k,:])#(2n,)
            bk = self.fit_b(X_nd,wei)
            _,pterr = self.find_t(X_nd,list(bk))# size(n,d)
            err2 = np.sum(pterr**2,axis=1)
            sigma2_K[k]=np.dot(wei[:n],err2)
            beta_Km[k,:]=bk
        return beta_Km,sigma2_K
        
    def E_step(self,lam_K,beta_Km,sigma2_K):
        pdf_Kn = self.pdf_z_given_beta(beta_Km,sigma2_K)
        w_Kn_tmp =(lam_K*pdf_Kn.T).T
        sum_w_n = np.sum(w_Kn_tmp,axis=0)
        w_Kn = w_Kn_tmp/sum_w_n
        w_Kn[:,sum_w_n==0]=0
        lam_K_new = np.mean(w_Kn,axis=1)
        logl = np.log(np.dot(lam_K_new[None,:],pdf_Kn))# size(1,n)
        logl = np.sum(logl)
#        lam_K_new = lam_K_new/np.sum(lam_K_new)
        return w_Kn,lam_K_new,logl
    def run(self,tol=1e-4,MaxIt=1000):
        
        sigma2_K = self.sigma2_ini        
        lam_K = self.lam_ini
        beta_Km = self.beta_ini
        beta_path = beta_Km
        lam_path = lam_K
        sigma2_path = sigma2_K
        
        er=100
        it=0
        w_Kn,lam_K,logl   = self.E_step(lam_K,beta_Km,sigma2_K)
        logl_path=logl
        while er>tol:            
            beta_Km,sigma2_K = self.M_step(w_Kn)
            w_Kn,lam_K,logl   = self.E_step(lam_K,beta_Km,sigma2_K)
            # recording path
            beta_path = np.dstack((beta_path,beta_Km))
            lam_path = np.vstack((lam_path,lam_K))
            sigma2_path = np.vstack((sigma2_path,sigma2_K))
            logl_path = np.vstack((logl_path,logl))
            # calculate error
            er = np.abs(logl_path[-1]-logl_path[-2])
            print('Iteration: %d, logl=%2.4f, err= %2.4f'  % (it,logl,er))
            it=it+1
        self.result = {'beta_Km':beta_Km,'sigma2_K':sigma2_K,'lam_K':lam_K}
        self.path = {'beta':beta_path,'lam':lam_path,'sigma2':sigma2_path,'logl':logl_path}
    def plot_result2D(self,MarkerSize = 50,savename=None):
        beta_path = self.path['beta']
        tt = np.arange(-15,15,1e-1)
        X = self.data
        fig = plt.figure(num=None, figsize=[5,5], dpi=80)
        ax = fig.add_subplot(111)
        ax.scatter(X[:,0],X[:,1],s=MarkerSize)
        for k in range(self.K):
            xx,yy = self.f(tt,*list(beta_path[k,:,-1]))
            ax.plot(xx,yy)
        ax.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
        ax.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
        ax.set_xticks([], []);ax.set_yticks([], []);
        ax.set_title('EM algorithm results')
        if savename is not None:
            fig.tight_layout()
            plt.savefig(savename+'.png')
  
class EM_line_model(EM_cluster):
    
    def __init__(self,data,parms,surface_dim=1,result=None):
        self.surface_dim = surface_dim
        EM_cluster.__init__(self,data,parms,line,line_ini)
        # Although I defined function line here, the program never use it.
        
    def fit_b(self,X_nd,wei):
        
        n,d = X_nd.shape
        wi = wei[:n]
        K = self.surface_dim
        bparm = np.zeros(d+K*d)
        
        wi = wi[...,None]
        wi = wi/np.sum(wi)
#        if np.sum(wi_other>0)>d-1:#need at least d-1 points
        Xbari = np.dot(wi.T,X_nd)#1-by-d
        bparm[:d]=Xbari
        Bi = X_nd-Xbari
        BwBi = np.dot(Bi.T,(Bi*wi))
        D,V = np.linalg.eig(BwBi)
        ind = np.argsort(D)#sort from small to big
        for k in np.arange(K):
            v = V[:,ind[-k-1]]# the biggest eigenvector
            v = v[...,None]
            bparm[d+k*d:d+(k+1)*d]=v.flatten()*np.sign(v[0])
        return bparm
        
    def err_fun(self,bparm,X_nd):
        X_nd = X_nd.astype('float64')
        n,d = X_nd.shape
        Xbari = bparm[:d]
        vec_nd= X_nd-Xbari
        u = bparm[d:]
        K = self.surface_dim
        Eye = np.eye(d)
        err = np.zeros((n,d))
        for i in range(n):
            veci = vec_nd[i,:]
            veci = veci[None,...]
            mat = np.zeros((d,d))
            for k in range(K):
                v = u[k*d:(k+1)*d]
                v = v[...,None]
                mat +=v*v.T
            err[i,:] = np.dot(veci,Eye-mat)#%% size(2,1)
        err_flat = err.flatten('F')
        return err_flat
    def fnp(self,t,*arg):
        d = self.dim
        K = self.surface_dim
        xbari = arg[:d]
        bp = arg[d:]
        bp = np.reshape(bp,(K,d))
        vec = np.sum(bp,axis=0)
        y = xbari+t[:,None]*vec[None,:]
        return y.T
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
def Linf_norm(X):
    absx = np.abs(X)
    tmp = absx.flatten(1)
    return np.max(tmp)