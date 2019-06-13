# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 09:11:02 2017

@author: Wayne
"""

#%%
from importlib import reload
import RRclustering
import Aux_fcn

import matplotlib.pyplot as plt
import numpy as np
reload(RRclustering)
reload(Aux_fcn)
from Aux_fcn import plot_result_line,plot_thr_result_files,plot_result3Dplane,plot_result3Dplane_with_exact
from RRclustering import RRalg,save_result,load_result

import gen_data

reload(gen_data)


###############################################################################################
#%% 2D line example: 
    #a. Cross data: 
    #%%1. generate, run
n=50
data = gen_data.gen_cross(n)
parms = {'beta':0.8,'lambda':100,'Model_Type':'line','initial_method':'Local_Best'}
a=RRalg(data,parms)
a.run()
Aux_fcn.plot_result(a,MarkerSize=50)
#%% the exact sol
fv_l, Zid = Aux_fcn.run_exactsol_with_C(a,m=2)
#%%
Aux_fcn.plot_result(a,Zid_exact=Zid,MarkerSize=50)
###############################################################################################
#%% b. Scurve data: 
    #%%1. generate, run
n=50
data,_=gen_data.Scurve(n,True)
parms = {'beta':0.8,'lambda':70,'Model_Type':'line','initial_method':'Local_Best'}
a=RRalg(data,parms)
a.run()
Aux_fcn.plot_result(a,MarkerSize=50)

#%% the exact sol
fv_l, Zid = Aux_fcn.run_exactsol_with_C(a,m=3)
# m=3: Zid = [ 601 1004 1223]
#%%
Aux_fcn.plot_result(a,Zid_exact=Zid,MarkerSize=50)
###############################################################################################
#%% 3D line example:
#%% line3D
lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,1.],[1.,2.,5.,-2.,3.,4.],[1.,1.,1.,1.,10.,3.]]}
lib_parms['ls_range_x'] = [[-3,3],[-1,4],[0,5]]
lib_parms['ls_var_yz'] = [.3,.3,.3]
n=50
X = gen_data.lines3D(n,lib_parms)
gen_data.plotpt(X,title='Original data')

parms = {'beta':0.8,'lambda':50,'Model_Type':'line'}
parms['initial_method']='Random' # or 'Local_Best'
#%% run the algorithm
a = RRalg(X,parms)
a.run()

#%% the exact solution
#lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,1.],[1.,2.,5.,-2.,3.,4.],[1.,1.,1.,1.,10.,3.]]}
#lib_parms['ls_range_x'] = [[-3,3],[-1,4],[0,5]]
#lib_parms['ls_var_yz'] = [.3,.3,.3]
#n=50
#X = gen_data.lines3D(n,lib_parms)
fv_l, Zid = Aux_fcn.run_exactsol_with_C(a,m=3)
# m=1: h = 14.921015730101239, Zid = [1105]
# m=2: h = 7.6151797697776678, Zid = [129,879]
# m=3: h = 4.9871628878148089, Zid = [84, 879, 1129]# 11 sec
#%%
Aux_fcn.plot_result(a,Zid_exact = [84,  879, 1129],MarkerSize=50,savename = 'tmp')


###############################################################################################
#%% 3D plane example
#%%
lib_parms={'ls_plane_parm':[[1,1,1,0],[1,-1,1,0],[0,0,1,-2]]}
lib_parms['ls_range_xy']=[[0,5,-5,5],[0,5,-5,5],[0,5,0,5]] 
lib_parms['ls_var_z']=[.3,.3,.3]
X = gen_data.planes3D(40,lib_parms)
gen_data.plotpt(X)
#%%
parms = {'beta':0.8,'lambda':50,'Model_Type':'plane','initial_method':'Local_Best'}
a = RRalg(X,parms)
a.run()
#%%
Aux_fcn.run_exactsol_with_C(a,m=2) 
#%% exact solution
#m=1: h=10.116827595188438, Zid = 3495
#m=2: h=5.2251117315217863, Zid = [ 751, 7653] 3.73 sec
#m=3: h=2.5817415541183859, Zid = [[1618, 6945, 7653]] 5138.45 sec
#%%
Aux_fcn.plot_result(a,Zid_exact =[1618, 6945, 7653],MarkerSize=50,elev = -10,azim=150,savename='eg_plane_b0.8_l50_LocalBest_1')
#%%
Aux_fcn.plot_result(a,Zid_exact =[1618, 6945, 7653],MarkerSize=50,elev = -30,azim=-5)
#%%
Aux_fcn.plot_result(a,MarkerSize=50)
