# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:19:41 2017

@author: wayin
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
#%% Scurve
data_name = 'Scurve'
X,_=gen_data.Scurve(50,True)
gen_data.plotpt(X,ticks=False,savename='result_plots\M2R2\\'+data_name)
n = X.shape[0]
#%%
parms={'beta':0.8,'lambda':80,\
       'list_NumPt4Instance':[2],\
       'dist_adj':True,'initial_method':'Local_Best',\
       'ini_seed':3}
       #%%
for lam in [10,20,110,120]:
    parms['lambda']=lam    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M2R2\\'+data_name+'_n%d_b%1.1f_l%d' \
             %(n,parms['beta'],parms['lambda'])
    save_result(a,name)
#%%
lam_set =  [10,20,110,120]
files = ['result_rra\M2R2\\'+data_name+'_n%d_b0.8_l%d' %(n,lam) for lam in lam_set]
#%%
name = 'result_plots\M2R2\\'+data_name +'_RRA_'+parms['initial_method']
plot_thr_result_files(files,figsize=(20,5),layout=[1,4],savename = name)
#%% Different seed
parms['lambda']=110
#%%
for seed in range(6):
    parms['ini_seed']=seed    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M2R2\\'+data_name+'_n%d_b%1.1f_l%d_seed%d' \
             %(n,parms['beta'],parms['lambda'],seed)
    save_result(a,name)
#%%
seed_set =  np.arange(6)
files = ['result_rra\M2R2\\'+data_name+'_n%d_b0.8_l%d_seed%d' %(n,parms['lambda'],seed) for seed in seed_set]
name = 'result_plots\M2R2\\'+data_name +'_RRA_'+'Random'
plot_thr_result_files(files,figsize=(30,5),layout=[1,6],savename = name)
#%% Exact Sol  
for m in [2,3]:
    _, Zid = a.run_exactsol_with_C(m)
    name = 'result_plots\M2R2\\'+data_name+'_exact_m%d' %m
    plot_result(a,Zid,savename=name)    
    
