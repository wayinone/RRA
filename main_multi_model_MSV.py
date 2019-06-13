# -*- coding: utf-8 -*-
"""
Created on Thu May  4 06:42:55 2017

@author: Wayne
"""
import matplotlib.pyplot as plt
from importlib import reload
import RRclustering3
import plot_tool
import minSumVol
import gen_data
import numpy as np
from RRclustering3 import RRalg
import MSVolume
import plot_tool_MSV
reload(plot_tool_MSV)
reload(MSVolume)
reload(RRclustering3)
reload(plot_tool)
reload(gen_data)
reload(minSumVol)

#%% Example 1.0: clean percent sign in 2D %%%%%%%%%%%%%%%%%
data_name='pct clean'
lib_parms={'ls_pts':[[-2,4],[2,-2]],'var':[[0.1],[0.1]]}
X0 = gen_data.points(50,lib_parms)
lib_parms={'ls_line_parm': [[0.,0.,1.,1.]]}
lib_parms['ls_range_x'] = [[-3,5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines2D(40,lib_parms)
X = np.vstack((X0,X1))
gen_data.plotpt(X)
#%%
#%% Example 1.1: corruted percent sign in 2D %%%%%%%%%%%%%%%%%
data_name='pct'
lib_parms={'ls_pts':[[-2,4],[2,-2]],'var':[[0.1],[0.1]]}
X0 = gen_data.points(50,lib_parms)
lib_parms={'ls_line_parm': [[0.,0.,1.,1.]]}
lib_parms['ls_range_x'] = [[-3,5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines2D(40,lib_parms)
X2 = np.vstack((X0,X1))
gen_data.plotpt(X2)
X = gen_data.add_outlier(40,X2)
#%%#%% Example 2.0: points3D
data_name='3Dmix_clean'
lib_parms={'ls_pts':[[4,4,5]],'var':[[0.1]]}
X0 = gen_data.points(15,lib_parms)
##%% line3D
lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,10.]]}
lib_parms['ls_range_x'] = [[-3,2]]
lib_parms['ls_var_yz'] = [0.1]
X1 = gen_data.lines3D(15,lib_parms)
##%% plane3D
lib_parms={'ls_plane_parm':[[0,0,1,10]]}
lib_parms['ls_range_xy']=[[-3,5,-3,5]] 
lib_parms['ls_var_z']=[.05]
X2 = gen_data.planes3D(25,lib_parms)
X = np.vstack((X0,X1,X2))
#%% Example 2.1: points3D
data_name='3Dmix'
lib_parms={'ls_pts':[[4,4,5]],'var':[[0.1]]}
X0 = gen_data.points(15,lib_parms)
##%% line3D
lib_parms={'ls_line_parm': [[0.,0.,0.,5.,-4.,10.]]}
lib_parms['ls_range_x'] = [[-3,2]]
lib_parms['ls_var_yz'] = [0.1]
X1 = gen_data.lines3D(15,lib_parms)
##%% plane3D
lib_parms={'ls_plane_parm':[[0,0,1,10]]}
lib_parms['ls_range_xy']=[[-3,5,-3,5]] 
lib_parms['ls_var_z']=[.05]
X2 = gen_data.planes3D(25,lib_parms)
X3 = np.vstack((X0,X1,X2))
X = gen_data.add_outlier(10,X3)
#%%
n = X.shape[0]
folder = 'result_plots\mix_model_results'
savename=folder+'\MV_'+data_name+'_n%d'  % n
gen_data.plotpt(X,title=data_name,ticks=True,savename=savename)
gen_data.plotpt(X,title=data_name,savename=savename+'_2',azim=161,elev=0)

#%%
parms = {'beta':0.5,'lambda':70}
b = MSVolume.MSV(X,parms)
#%%
m=3
percentile=0.7
b.run(m=m,percentile=percentile)
#c = b.dict_msv
#%% candidates
candname=savename+'candidates'
plot_tool_MSV.plot_result(b, b.candidates,plot_volume=False,title_add=' candidates',\
                          ticks = True,savename=candname)
#%% show volumes
resultname=savename+'_%dth_ptile_m%d' %(percentile*100,m)
plot_tool_MSV.plot_result(b,b.id_found,ticks=True,\
                          title_add=' %dth pct' %(percentile*100),savename=resultname)
#%% only show instances, no volume showed
result1name=savename+'_%dth_ptile_m%d_no_vol' %(percentile*100,m)
plot_tool_MSV.plot_result(b,b.id_found,ticks=True,\
                          plot_volume=False,\
                          title_add=' %dth pct' %(percentile*100),savename=result1name)
#%%
plot_tool_MSV.plot_result(b,b.id_found,ticks=True,\
                          plot_volume=False,\
                          title_add=' %dth pct' %(percentile*100),\
                          azim=161,elev=0,savename=result1name+'_2')
plot_tool_MSV.plot_result(b,b.id_found,ticks=True,\
                          title_add=' %dth pct' %(percentile*100),savename=resultname+'_2'\
,azim=161,elev=0)
#%% for test
#lid = 18
#plot_tool_MSV.plot_result(b,c['l_id_list'][lid],title_add='%1.2f' % c['Vols'][lid])
