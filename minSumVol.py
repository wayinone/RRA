# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 07:07:04 2017

@author: Wayne
"""
#%%
import numpy as np
import math

import itertools as it
import warnings
#%%

def cluster_MinVol(a,list_Allz_M,m,percentile=1):
    """
    percentile: a value in [0,1]
    """
    data = a.data
    beta = a.beta
    M = len(list_Allz_M)
    tmp = it.combinations(np.arange(0,M),m)
    AllComb = np.vstack(tmp) #(Ni,p-1)
    #%%
    n_comb = AllComb.shape[0]
    Vols = np.zeros(n_comb)
    l_id_list = [[list_Allz_M[AllComb[l,i]] for i in range(m)] for l in range(n_comb)]
    list_r_in = [None]*n_comb
    list_r_dist = [None]*n_comb
#
#%%
    for l in range(n_comb):
        l_id = [list_Allz_M[AllComb[l,i]] for i in range(m)]
        p = len(l_id)
        try:    
            cc,_ = a.fcn_classify(l_id)
        except:
            warnings.warn('list of zeta out of range, change zeta =1')
            cc,_ = a.fcn_classify(l_id)
        Vols[l],_,list_r_in[l],list_r_dist[l]=Volumn_cc(data,cc,l_id,beta,percentile)
    
    id_min = np.argmin(Vols)
    
    list_xid_found =[list_Allz_M[AllComb[id_min,i]] for i in range(m)]
    
    parms={'list_r_in':list_r_in,'list_r_dist':list_r_dist}
    return list_xid_found,l_id_list,id_min,Vols,parms

def Volumn_cc(data,cc,l_id,beta,percentile):
    """
    
    """
    def BallVol(radius,dim):
        return (np.pi)**(dim/2)/math.gamma(dim/2+1)*radius**dim
    m = len(np.unique(cc))
    n,d = data.shape
    Vols = np.zeros(m)
    r_in = np.zeros(m)
    r_out = np.zeros(m)
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
                r_dist_arr = norm(Y)**beta
                r_inclass=1.0
            if pi>1:
                 l_id_i = l_id[i][1:]
                 Y_p = data[l_id_i,:]-X0#(p-1,d)
                 A = Y_p.T#(d,p-1)
                 AtAinv = np.linalg.inv(np.dot(A.T,A))#(p-1,p-1)              
                 AtAinv_At = np.dot(AtAinv,A.T)
                 M_dd = np.dot(A,AtAinv_At)#(d,d)
                 proj_dn = np.dot(M_dd,Y.T)
                 I_proj_dn = Y.T-proj_dn
                 r_dist_arr = norm(I_proj_dn.T)**beta
    
                 
                 x = proj_dn.T
                 mu = np.mean(x,axis=0)
                 r_inclass_arr=norm(x-mu)**beta
                 r_inclass = np.max(r_inclass_arr)
        
    #        r_dist = np.mean(r_dist_arr)
            r_dist = np.percentile(r_dist_arr,percentile*100)
            r_in[i]=r_inclass
            r_out[i]=r_dist
            Vols[i] =BallVol(r_inclass,pi-1)*BallVol(r_dist,d-(pi-1))
    fval = np.sum(Vols)
    return fval,Vols,r_in,r_out
def norm(Xnd):
    c = np.sqrt(np.sum(Xnd**2,axis=1))
    return c