# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:03:34 2017

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
data_name = 'lines3D'
lib_parms={'ls_line_parm': [[0.,0.,3.,5.,-4.,1.],\
                            [1.,2.,5.,-2.,3.,4.],\
                            [1.,1.,1.,1.,10.,3.]]}
lib_parms['ls_range_x'] = [[-3,5],[-1,4],[0,4]]
lib_parms['ls_var_yz'] = [.3,.3,.3]
n=60
X = gen_data.lines3D(n,lib_parms)
name = 'result_plots\M2R3\\'+data_name+'wo_ticks'
gen_data.plotpt(X,ticks=False,savename=name)
#%%
gen_data.plotpt(X,ticks=False,savename=name,elev=24,azim=5)

##%%
#%%
n = X.shape[0]
parms={'beta':0.8,'lambda':100,\
       'list_NumPt4Instance':[2],\
       'dist_adj':True,'initial_method':'Local_Best',}
a=RRalg(X,parms)
#%%
a.run()
#%%
name = 'result_plots\M2R3\\'+data_name+'_n%d_b0.8_l%d_2' %(n,parms['lambda'])
plot_result(a,a.list_xid_found,savename=name,elev=24,azim=5)

#%%
##%%
#for lam in [30,40]:
#    parms['lambda']=lam    
#    a=RRalg(X,parms)
#    a.run()
#    name = 'result_rra\M2R3\\'+data_name+'_n%d_b%1.1f_l%d' \
#             %(n,parms['beta'],parms['lambda'])
#    save_result(a,name)
#%%
lam_set =  [20,30,250,260]
files = ['result_rra\M2R3\\'+data_name+'_n%d_b0.8_l%d' %(n,lam) for lam in lam_set]
name = 'result_plots\M2R3\\'+data_name+'_RRA_0'
plot_thr_result_files(files,figsize=(20,5),layout=[1,4],savename = name)

#%%
plot_result(a,a.list_)
#%% Exact Sol  
for m in [2,3]:
    _, Zid = a.run_exactsol_with_C(m)
    name = 'result_plots\M2R3\\'+data_name+'_exact_m%d_1' %m
    plot_result(a,Zid,title='exact',savename=name,elev=24,azim=5)
