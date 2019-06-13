# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 07:49:38 2017

@author: Wayne
"""
#%%
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
    #a. triangle: 
    #%%1. generate, run
data_name = 'Triangle'
lib_parms={'ls_line_parm': [[-1.,0.,1,0.],[-1.,0.,1.,2*np.sqrt(3)],[1.,0.,-1.,2*np.sqrt(3)]]}
lib_parms['ls_range_x'] = [[-1.5,1.5],[-1.5,0.5],[-0.5,1.5]]
lib_parms['ls_var_yz'] = [.001,.001,.001]
lib_parms['n_outlier'] = 0
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]

X = gen_data.lines2D(70,lib_parms)
X = gen_data.add_outlier(15,X)
gen_data.plotpt(X)
#%%
gen_data.plotpt(X,savename='result_plots\M2R2\\'+data_name)
#%%
parms={'beta':0.8,'lambda':70,\
       'list_NumPt4Instance':[2],\
       'dist_adj':True,'initial_method':'Random'}

a=RRalg(X,parms)
a.run()
plot_result(a,a.list_xid_found)
#%%
a.run()
#%%
for lam in [20,30,40,50]:
    parms={'beta':0.8,'lambda':lam,\
           'list_NumPt4Instance':[2],\
           'dist_adj':True,'initial_method':'Local_Best'}
    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M2R2\\'+data_name+'_n%d_b%1.1f_l%d' \
             %(n,parms['beta'],parms['lambda'])
    save_result(a,name)
#%%
lam_set = [30,40,140,150]
files = ['result_rra\M2R2\\'+data_name+'_n60_b0.8_l%d' %lam for lam in lam_set]
#%%
plot_thr_result_files(files,figsize=(20,5),layout=[1,4])
#%% the exact sol
for m in [1,2]:
    _, Zid = a.run_exactsol_with_C(m)
    name = 'result_plots\M2R2\\'+data_name+'_exact_m%d' %m
    plot_result(a,Zid,savename=name)

   #a. cor Xdata: 
    #%%1. generate, run
data_name = 'Xshape_cor'
lib_parms={'ls_line_parm': [[0.,0.,5.,-4.],[0.,0.,1.,.5]]}
lib_parms['ls_range_x'] = [[-2,2],[-4,4]]
lib_parms['ls_var_yz'] = [.01,.01]
lib_parms['n_outlier'] = 40
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]
n=100
X = gen_data.lines2D(n,lib_parms)
#X = gen_data.add_outlier(20,X1,random_seed=3)
gen_data.plotpt(X,ticks=False,savename='result_plots\M2R2\\'+data_name)
n = X.shape[0]

#%%
for lam in [30,50]:
    parms={'beta':0.8,'lambda':lam,\
           'list_NumPt4Instance':[2],\
           'dist_adj':True,'initial_method':'Local_Best'}
    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M2R2\\'+data_name+'_n%d_b%1.1f_l%d' \
             %(n,parms['beta'],parms['lambda'])
    save_result(a,name)
#%%
lam_set = [50]
files = ['result_rra\M2R2\\'+data_name+'_n100_b0.8_l%d' %lam for lam in lam_set]
plot_thr_result_files(files,figsize=(5,5),layout=[1,1])
#%% the exact sol
for m in [1,2]:
    _, Zid = a.run_exactsol_with_C(m)
    name = 'result_plots\M2R2\\'+data_name+'_exact_m%d' %m
    plot_result(a,Zid,savename=name)
#%%
g = load_result(files[0])

#%%
fig = plt.figure(1)
#%%
fig = plt.figure(num=None, figsize=[10,5], dpi=80)
#%%
axe = fig.add_subplot(1,2,1)
gen_data.plotpt(X,ax=axe)

ax2 = fig.add_subplot(1,2,2)
plot_result(g,g.list_xid_found,ax=ax2)