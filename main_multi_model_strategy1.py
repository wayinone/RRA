# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 05:44:18 2017

@author: Wayne
"""


#%%
from importlib import reload
import RRclustering3
import Aux_fcn2
import plot_tool
import matplotlib.pyplot as plt
import numpy as np
reload(RRclustering3)
reload(Aux_fcn2)
reload(plot_tool)
from RRclustering3 import RRalg
import gen_data

reload(gen_data)
import wrapper_fcns
reload(wrapper_fcns)

import itertools as it
#%%
#%% points and line
lib_parms={'ls_pts':[[-2,4],[2,-2]],'var':[[0.1],[0.1]]}
X0 = gen_data.points(50,lib_parms)
lib_parms={'ls_line_parm': [[0.,0.,1.,1.]]}
lib_parms['ls_range_x'] = [[-3,5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines2D(40,lib_parms)
X2 = np.vstack((X0,X1))
gen_data.plotpt(X2,savename='clean_percentile_data')
#%%
X = gen_data.add_outlier(40,X2)
#gen_data.plotpt(X,savename='mix_model_results\percentile_data_cor_n130')
gen_data.plotpt(X)
#%%
parms = {'beta':0.8,'lambda':70,'list_NumPt4Instance':[1,2],\
         'disp_info':True,'list_dist_mply':[1,1.5]}
a=RRalg(X,parms)
a.run()
#%%
save_name = 'mix_model_results\stgy1_percentile_data_n90_%d' %parms['list_dist_mply'][1]
plot_tool.plot_result(a,a.list_xid_found,title='show_weight',\
                      savename = save_name)

#%% points and line 2
lib_parms={'ls_pts':[[5,4]],'var':[[0.15]]}
X0 = gen_data.points(30,lib_parms)

lib_parms={'ls_line_parm': [[0.,0.,1.,1.]]}
lib_parms['ls_range_x'] = [[-4,2.5]]
lib_parms['ls_var_yz'] = [0.05]
X1 = gen_data.lines2D(50,lib_parms)
X = np.vstack((X0,X1))
gen_data.plotpt(X)
#%%
parms = {'beta':0.8,'lambda':70,'list_NumPt4Instance':[1,2],\
         'disp_info':True,'list_dist_mply':[1,1.6]}
a=RRalg(X,parms)
a.run()
plot_tool.plot_result(a,a.list_xid_found,title='show_weight',savename = 'mix_model_results\stgy1_exclaimMark_data_n70')