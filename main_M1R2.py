# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:22:33 2017

@author: Wayne
"""

import sys
#sys.path.append("RRA")
from importlib import reload
import gen_data,plot_tool
import RRclustering3
reload(RRclustering3)
reload(gen_data)
reload(plot_tool)
from RRclustering3 import RRalg, save_result,load_result
from plot_tool import plot_result,plot_thr_result_files
import numpy as np
#%%

import matplotlib.pyplot as plt
from matplotlib import rc, colors
from sklearn.cluster import KMeans
#%%
#data_name = 'Dice3'
#lib_parms={'ls_pts':[[-1,1],[0,1.7],[1,1]],\
#          'var':[0.05]*3}
#X=gen_data.points(200,lib_parms)
#
#gen_data.plotpt(X,savename='result_plots\M1R2\\'+data_name)
#%%
data_name = 'Stars'
ss = [[i/2,0] for i in np.arange(0,3)]
ss=ss +[[i/2,1/2] for i in np.arange(0,3)]
ss=ss +[[i/2,1] for i in np.arange(0,3)]
lib_parms={'ls_pts':ss,\
           'var':[0.015]*9,\
           'distr':'Gaussian'}
X=gen_data.points(200,lib_parms)
#X = gen_data.add_outlier(100,X)
gen_data.plotpt(X,savename='result_plots\M1R2\\'+data_name)
#%%
data_name = 'Mouse'
lib_parms={'ls_pts':[[-.4,.4],[0,0],[.4,.4]],\
           'var':[0.02,0.2,0.02],\
           'ls_n':[2,8,2],\
            'distr':'Uniform'}
X=gen_data.points(200,lib_parms)
#X = gen_data.add_outlier(100,X)
gen_data.plotpt(X,savename='result_plots\M1R2\\'+data_name)
#%% k-means
kmeans = KMeans(n_clusters=9, random_state=0).fit(X)
cc = kmeans.labels_
colorname = list(colors.cnames.keys())
colorname.pop(2)
colorname = ['r','g','b','c','m','y','k']+colorname
clist = [colorname[cc[k]] for k in range(X.shape[0])]
plt.scatter(X[:,0],X[:,1],c=clist,cmap='jet')

#%%
n = X.shape[0]
parms={'beta':0.8,'lambda':100,\
       'list_NumPt4Instance':[1],\
       'dist_adj':True,'initial_method':'Random',\
       'ini_seed':3}
for lam in [50]:
    parms['lambda']=lam    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M1R2\\'+data_name+'_n%d_b%1.1f_l%d' \
             %(n,parms['beta'],parms['lambda'])
    save_result(a,name)
#%%
plot_result(a,a.list_xid_found)
#%%
lam_set =  [50,350]
files = ['result_rra\M1R2\\'+data_name+'_n%d_b0.8_l%d' %(n,lam) for lam in lam_set]
plot_thr_result_files(files,figsize=(20,5),layout=[1,4],savename='result_plots\M1R2\\'+data_name+'_RRA')
#%%
#%% Exact Sol  
for m in [2,3]:
    _, Zid = a.run_exactsol_with_C(m)
    name = 'result_plots\M1R2\\'+data_name+'_exact_m%d' %m
    plot_result(a,Zid,title='exact',savename=name)

    
    
    #%%
data_name = 'Dice5'
lib_parms={'ls_pts':[[-1,1],[-1,-1],[0,0],[1,-1],[1,1]],\
          'var':[0.01]*5}
X=gen_data.points(200,lib_parms)

gen_data.plotpt(X)
#%%
parms={'beta':0.8,'lambda':70,\
       'list_NumPt4Instance':[1],\
       'dist_adj':True,'initial_method':'Random',\
       'ini_seed':3}

parms['lambda']=lam    
a=RRalg(X,parms)
a.run()