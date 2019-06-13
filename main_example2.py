# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:15:57 2017

@author: Wayne
"""

from importlib import reload
import RRclustering2
import RRclustering3
import Aux_fcn2

import matplotlib.pyplot as plt
import numpy as np
reload(RRclustering2)
reload(RRclustering3)
reload(Aux_fcn2)
from Aux_fcn2 import plot_result_line,plot_thr_result_files,plot_result3Dplane,plot_result3Dplane_with_exact
from RRclustering3 import RRalg,save_result,load_result

import gen_data

reload(gen_data)
import wrapper_fcns
reload(wrapper_fcns)
#%% Scurve
n=80
X,_=gen_data.Scurve(n,True)
#%%
parms = {'beta':0.8,'lambda':70,'Model_Type':'line','initial_method':'Local_Best'}
a=RRalg(X,parms)
a.run()
#%%
plot_result(a,a.list_xid_found,MarkerSize=50)


###############################################################################################
#%% points3D
lib_parms={'ls_pts':[[-2,4,15]],'var':[[0.1]]}
n=10
X0 = gen_data.points(n,lib_parms)
#gen_data.plotpt(X0,title='Original data')

##%% line3D
lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,10.],[1.,1.,1.,-3.,-2.,1.]]}
lib_parms['ls_range_x'] = [[-3,2],[-4,3]]
lib_parms['ls_var_yz'] = [0.05,0.05]
n=30
X1 = gen_data.lines3D(n,lib_parms)
#gen_data.plotpt(X1,title='Original data')

##%% plane3D
lib_parms={'ls_plane_parm':[[0,0,1,10]]}

lib_parms['ls_range_xy']=[[0,8,0,8]] 
lib_parms['ls_var_z']=[.05]
X2 = gen_data.planes3D(20,lib_parms)
#gen_data.plotpt(X2,title='Original data')
##%%
X = np.vstack((X0,X1,X2))
gen_data.plotpt(X,title='Original data')
#%%
def custom3(x):
    return len(x)**2
parms = {'beta':0.8,'lambda':100,'list_NumPt4Instance':[1,2,3]}
parms['fcn_gamma']=custom3  
parms['initial_method']='Random' # or 'Local_Best'
parms['list_dist_mply'] = [1.,1.2,2.1]
#% run the algorithm
a = RRalg(X,parms)
#%%
a.run()
#%%
Aux_fcn2.plot_result(a,a.list_xid_found,title = parms['list_dist_mply']\
                     ,savename = 'mix_instance_b0.8_l100_2')
#%% the exact solution
#lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,1.],[1.,2.,5.,-2.,3.,4.],[1.,1.,1.,1.,10.,3.]]}
#lib_parms['ls_range_x'] = [[-3,3],[-1,4],[0,5]]
#lib_parms['ls_var_yz'] = [.3,.3,.3]
#n=50
#X = gen_data.lines3D(n,lib_parms)
fv_l, Zid = Aux_fcn2.run_exactsol_with_C(a,m=3)
# m=1: h = 14.921015730101239, Zid = [1105]
# m=2: h = 7.6151797697776678, Zid = [129,879]
# m=3: h = 4.9871628878148089, Zid = [84, 879, 1129]# 11 sec

Aux_fcn2.plot_result(a,Zid)
#%%
Aux_fcn2.plot_result(a,[[2,2,2,2,2,2],[0,1,2,3,4,5]])
#%%
Aux_fcn.plot_result(a,Zid_exact = [84,  879, 1129],MarkerSize=50,savename = 'tmp')


###############################################################################################
#%%
#%% points3D
lib_parms={'ls_pts':[[-2,4,15]],'var':[[0.1]]}
X0 = gen_data.points(10,lib_parms)
##%% line3D
lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,10.]]}
lib_parms['ls_range_x'] = [[-3,5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines3D(20,lib_parms)
##%% plane3D
lib_parms={'ls_plane_parm':[[0,0,1,10]]}
lib_parms['ls_range_xy']=[[0,5,0,5]] 
lib_parms['ls_var_z']=[.05]
X2 = gen_data.planes3D(20,lib_parms)
X = np.vstack((X0,X1,X2))
gen_data.plotpt(X,title='Original data')
#%%
def custom3(x):
    return len(x)**2
parms = {'beta':0.8,'lambda':100,'list_NumPt4Instance':[1,2,3]}
parms['fcn_gamma']=custom3  
parms['initial_method']='Random' # or 'Local_Best'
parms['list_dist_mply'] = [1.,1.2,2]
#% run the algorithm
a = RRalg(X,parms)
a.run()
Aux_fcn2.plot_result(a,a.list_xid_found)
#%%
    #%%1. generate, run
n=50
data,_=gen_data.Scurve(n,True)
parms = {'beta':0.8,'lambda':70,'list_NumPt4Instance':[1,2]}
a=RRalg(data,parms)
#%%
a.run()
#%%
Aux_fcn2.plot_result(a,a.list_xid_found,MarkerSize=50)
#%%
#%% points and line
lib_parms={'ls_pts':[[-2,4],[2,-2]],'var':[[0.1],[0.1]]}
X0 = gen_data.points(50,lib_parms)
lib_parms={'ls_line_parm': [[0.,0.,1.,1.]]}
lib_parms['ls_range_x'] = [[-3,5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines2D(30,lib_parms)
X = np.vstack((X0,X1))
gen_data.plotpt(X)
#%%
def custom(x):
    return len(x)**2
parms = {'beta':0.8,'lambda':70,'list_NumPt4Instance':[1,2]}
parms['fcn_gamma']=custom 
parms['list_dist_mply'] = [1.,1.5]
a=RRalg(X,parms)
a.run()
#%%
Aux_fcn2.plot_result(a,a.list_xid_found,MarkerSize=100)
#%%
