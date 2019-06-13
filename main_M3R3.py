# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:49:37 2017

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
data_name = 'plane 3D'
lib_parms={'ls_plane_parm':[[0,0,1,0],[0,1,1,0]]}
lib_parms['ls_range_xy']=[[0,5,-5,5],[0,5,-5,5]] 
lib_parms['ls_var_z']=[.2,.2,.2]
X = gen_data.planes3D(50,lib_parms)
name = 'result_plots\M3R3\\'+data_name+'_wo_tick_0'
gen_data.plotpt(X,ticks=False,savename=name,azim=18,elev=13)
name = 'result_plots\M3R3\\'+data_name+'_wo_tick_1'
gen_data.plotpt(X,ticks=False,savename=name,azim=-177,elev=0)
#%%
n = X.shape[0]
parms={'beta':0.8,'lambda':40,\
       'list_NumPt4Instance':[3],\
       'dist_adj':True,'initial_method':'Local_Best',}
a=RRalg(X,parms)
#%%
a.run()
#%%
name = 'result_plots\M3R3\\'+data_name+'_n%d_b%1.1f_l%d_0' \
             %(n,parms['beta'],parms['lambda'])
plot_result(a,a.list_xid_found,savename=name,elev=18,azim=13)
name = 'result_plots\M3R3\\'+data_name+'_n%d_b%1.1f_l%d_1' \
             %(n,parms['beta'],parms['lambda'])
plot_result(a,a.list_xid_found,savename=name,elev=-177,azim=0)             
#%%
for lam in [5,40,50,60]:
    parms['lambda']=lam    
    a=RRalg(X,parms)
    a.run()
    name = 'result_rra\M3R3\\'+data_name+'_n%d_b%1.1f_l%d' \
             %(n,parms['beta'],parms['lambda'])
    save_result(a,name)
#%%
lam_set =  [5,10,50,60]
files = ['result_rra\M3R3\\'+data_name+'_n%d_b0.8_l%d' %(n,lam) for lam in lam_set]
         #%%
name = 'result_plots\M3R3\\'+data_name+'_RRA_0'
plot_thr_result_files(files,figsize=(20,5),layout=[1,4],savename = name,azim=18,elev=13)
#%%
name = 'result_plots\M3R3\\'+data_name+'_RRA_1'
plot_thr_result_files(files,figsize=(20,5),layout=[1,4],savename = name,azim=-177,elev=0)
#%%
 _, Zid = a.run_exactsol_with_C(2)
 #%%
name = 'result_plots\M3R3\\'+data_name+'_exact_m2_0' 
plot_result(a,Zid,title='exact',savename=name,azim=18,elev=13)
name = 'result_plots\M3R3\\'+data_name+'_exact_m2_1' 
plot_result(a,Zid,title='exact',savename=name,azim=-177,elev=0)