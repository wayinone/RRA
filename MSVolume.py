# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:34:51 2017

@author: wayin
"""
#%%
import matplotlib.pyplot as plt

import RRclustering3
import plot_tool
import gen_data
import numpy as np
from RRclustering3 import RRalg
import numpy as np
import math

import itertools as it
import warnings
#%%

class MSV(object):
    """
    Minimum mean minimum sum of volumes
    Find first local minimum of mean sum of volumes:
        mean sum of volumes: MSV(m)/m
    """
    def __init__(self,X_nd, parms):
        self.X_nd = X_nd
        self.beta = parms['beta']
        parms['disp_info']=False
        self.parms = parms
        X_nd = self.X_nd
        d = X_nd.shape[1]
        param = self.parms
        for dd in range(d):
            p = dd+1
            param['list_NumPt4Instance']=[p]
            a=RRalg(X_nd,param)
            a.run()
            try:
                list_AllZ_M=list_AllZ_M+a.list_xid_found
            except NameError:
                list_AllZ_M=a.list_xid_found
        self.candidates = list_AllZ_M
        self.rra = a       
    def run(self,m=None,percentile=0.8): 
        if m is None:
            conti=1
            m=1
            ls_id_found = [None]*10
            ls_pms_msv = [None]*10
            prev_msv=1e10
            while conti:
                ls_id_found[m],pms= self.__cluster_MinVol(self.candidates,m,percentile)
                msv_m = pms['MMSV']
                ls_pms_msv[m]=pms
                if prev_msv/msv_m>1.5:
                    conti=0
                    best_m_found=m-1
                    print('Found %d instances with MMMSE' % best_m_found)
                else:
                    prev_msv = msv_m
                    m+=1
                if m==10:
                    conti=0
                    print('This method does not work!')
            id_FDMSV = ls_id_found[best_m_found] #first drastic decresing
            self.id_found = id_FDMSV
            self.ls_id_found=ls_id_found
            self.dict_msv=ls_pms_msv[best_m_found]
            self.best_m_found = best_m_found
        else:
            id_found,pms= self.__cluster_MinVol(self.candidates,m,percentile)
            self.dict_msv = pms
            self.id_found = id_found
            
    def __cluster_MinVol(self,list_Allz_M,m,percentile):
        """
        percentile: a value in [0,1]
        """
        data = self.X_nd
        beta = self.beta
        M = len(list_Allz_M)
        if m==M:
            AllComb = np.array([[0]])
        elif m>1:
            tmp = it.combinations(np.arange(0,M),m)
            AllComb = np.vstack(tmp) #(Ni,p-1)
        else:
            AllComb = np.arange(0,M)[:,None]
        #%%
        n_comb = AllComb.shape[0]
        Vols = np.zeros(n_comb)
        if m<M:
            l_id_list = [[list_Allz_M[AllComb[l,i]] for i in range(m)] for l in range(n_comb)]
        else:
            l_id_list = [list_Allz_M]
        list_r_in = [None]*n_comb
        list_r_dist = [None]*n_comb
        list_mu_in = [None]*n_comb
        list_vols =  [None]*n_comb
    #
    #%%
        for l in range(n_comb):
            l_id = [list_Allz_M[AllComb[l,i]] for i in range(m)]
            p = len(l_id)
            try:    
                cc= self.fcn_classify(l_id)
            except:
                warnings.warn('list of zeta out of range, change zeta =1')
                cc= self.fcn_classify(l_id)
            Vols[l],list_vols[l],list_r_in[l],list_r_dist[l],list_mu_in[l]=self.__Volumn_cc(cc,l_id,beta,percentile)
        
        id_min = np.argmin(Vols)
        
        list_xid_found =[list_Allz_M[AllComb[id_min,i]] for i in range(m)]
        
        parms_msv={'list_r_in':list_r_in,\
               'list_r_dist':list_r_dist,\
               'Vols':Vols,\
               'l_id_list':l_id_list,\
               'list_vols':list_vols,\
               'MSV':Vols[id_min],\
               'r_in_msv':list_r_in[id_min],\
               'mu_in_msv':list_mu_in[id_min],\
               'r_dist_msv':list_r_dist[id_min]}
        return list_xid_found,parms_msv

    def __Volumn_cc(self,cc,l_id,beta,percentile):
        """
        calculate the volumn using
        r_dist: max d(z,ell)^beta, for z in the class of ell
        r_inclass: max d(z-mu)^beta, for z in the class of ell
        """
        data = self.X_nd
        m = len(np.unique(cc))
        n,d = data.shape
        Vols = np.zeros(m)
        r_in = np.zeros(m)
        r_out = np.zeros(m)
        mu_in = np.zeros((m,d))
        for i in range(m):
            data_i = data[cc==i,:] #(k,d)
            k = data_i.shape[0]
            pi = len(l_id[i])            
            X0 = data[l_id[i][0],:]            
            Y = data_i-X0
            if k<n/15:
                Vols[i]=100*np.max(np.max(data,axis=0)-np.min(data,axis=0))*m# just some big values
            else:
                if pi==1:
        #            mu = np.mean(data_i,axis=0)
                    r_dist_arr = norm(Y)
                    r_inclass=1.0
                    mu = X0
                    mu_in_ = X0
                if pi>1:
                     l_id_i = l_id[i][1:]
                     Y_p = data[l_id_i,:]-X0#(p-1,d)
                     A = Y_p.T#(d,p-1)
                     AtAinv = np.linalg.inv(np.dot(A.T,A))#(p-1,p-1)              
                     AtAinv_At = np.dot(AtAinv,A.T)
                     M_dd = np.dot(A,AtAinv_At)#(d,d)
                     proj_dn = np.dot(M_dd,Y.T)
                     I_proj_dn = Y.T-proj_dn
                     r_dist_arr = norm(I_proj_dn.T)
        
                     
                     x = proj_dn.T#(n,d)
#                     id1,id2,d = get_max_dist(x)
#                     mu = x[id1,:]+x[id2,:]
                     
                     mu = np.mean(x,axis=0)
                     r_inclass_arr=norm(x-mu)
                     r_inclass = np.max(r_inclass_arr)
                     mu_in_ = mu+X0
            
        #        r_dist = np.mean(r_dist_arr)
                r_dist = np.percentile(r_dist_arr,percentile*100)
                r_inclass = r_inclass
                r_dist = r_dist
                mu_in[i,:]=mu_in_
                r_in[i]=r_inclass
                r_out[i]=r_dist
                Vols[i] =BallVol(r_inclass,pi-1)*BallVol(r_dist,d-(pi-1))
        fval = np.sum(Vols)
        return fval,Vols,r_in,r_out,mu_in
    def fcn_classify(self,list_xid_found):
        Xnd = self.X_nd
#        _,data_se = standerdizing(Xnd)
        n,d = Xnd.shape
        m = len(list_xid_found)
        distmat = np.zeros((n,m))
        for i in range(m):
            Z_ids = list_xid_found[i]# Z_ids is (p,) array, array of indexes of data
            p = len(Z_ids)
            distmat[:,i] = gen_dist_arr(Xnd,Z_ids,self.beta)
        data_cluster_id = np.argmin(distmat,axis=1)
#        h = np.sum(np.min(distmat,axis=1))
#        h_adj = h/(data_se**self.beta)
        return data_cluster_id
    def plot_best(self):
        plot_tool.plot_result(self.rra,self.id_found,title=None)#,savename=save_name)        
        plt.axis('equal')
def gen_dist_arr(Xnd,ids,beta,adj_quotient=1):
    d = Xnd.shape[1]
    Xi = Xnd[ids[0],:]
    Yi_dn = (Xnd-Xi).T #(d,n)
    p = len(ids)
    if p==1:
        dist = np.sqrt(np.sum(Yi_dn**2,axis=0))**beta #(n,)
        distarr = dist/adj_quotient
    else:
        A = Yi_dn[:,ids[1:]]#(d,p-1)
        AtAinv = np.linalg.inv(np.dot(A.T,A))#(2,2)
        AtAinv_At = np.dot(AtAinv,A.T)
        M_dd = np.eye(d)-np.dot(A,AtAinv_At)#(d,d)
        proj_dn = np.dot(M_dd,Yi_dn)
        dist = np.sqrt(np.sum(proj_dn**2,axis=0))**beta#(n,)
        distarr = dist/adj_quotient
    return distarr
def norm(Xnd):
    c = np.sqrt(np.sum(Xnd**2,axis=1))
    return c
def standerdizing(Y_nd):
    mean = np.mean(Y_nd,axis=0)
    tmp=np.var(Y_nd,axis=0)
    data_se = np.sqrt(np.sum(tmp))
    X_nd=(Y_nd-mean)/data_se
    return X_nd,data_se

def BallVol(radius,dim):
    return (np.pi)**(dim/2)/math.gamma(dim/2+1)*radius**dim

def get_max_dist(Y_nd):
    dist_mat_nn = dist_square(Y_nd)
    rc = np.where(dist_mat_nn == np.max(dist_mat_nn))
    r = rc[0][0];c=rc[1][0]
    maxd = np.sqrt(dist_mat_nn[r,c])
    return r,c,maxd
def dist_square(X_nd):
    n,d = X_nd.shape
    dist=np.zeros((n,n))
    for i in range(d):
        z = X_nd[:,i]
        z = z[...,None]
        zz = z*z.T
        zdiag=np.diag(zz)
        z2 = zdiag[...,None]*np.ones([1,n])
        disti = z2+z2.T-2*zz
        dist = dist+disti        
        
    dist[np.nonzero(dist<0)]=0
    return dist
