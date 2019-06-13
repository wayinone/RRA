# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:03:42 2017

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
#%% 'Scurve_hvytail'
data_name='ScurveHvy'
X = gen_data.Scurve_hvytail(50,dof=2.6)

#%% 'Triangle'
data_name = 'Triangle'
lib_parms={'ls_line_parm': [[-1.,0.,1,0.],[-1.,0.,1.,2*np.sqrt(3)],[1.,0.,-1.,2*np.sqrt(3)]]}
lib_parms['ls_range_x'] = [[-1.5,1.5],[-1.5,0.5],[-0.5,1.5]]
lib_parms['ls_var_yz'] = [.04,.04,.04]
lib_parms['n_outlier'] = 0
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]

X = gen_data.lines2D(70,lib_parms)
#%% 'Triangle_Cor'
data_name = 'Triangle_Cor'
lib_parms={'ls_line_parm': [[-1.,0.,1,0.],[-1.,0.,1.,2*np.sqrt(3)],[1.,0.,-1.,2*np.sqrt(3)]]}
lib_parms['ls_range_x'] = [[-1.5,1.5],[-1.5,0.5],[-0.5,1.5]]
lib_parms['ls_var_yz'] = [.04,.04,.04]
lib_parms['n_outlier'] = 0
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]

X1 = gen_data.lines2D(70,lib_parms)
X = gen_data.add_outlier(15,X1,random_seed=3) 
#%%  'Xshape_cor'
data_name = 'XshapeCor'
lib_parms={'ls_line_parm': [[0.,0.,5.,-4.],[0.,0.,1.,.5]]}
lib_parms['ls_range_x'] = [[-2,2],[-4,4]]
lib_parms['ls_var_yz'] = [.01,.01]
lib_parms['n_outlier'] = 40
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]
n=100
X = gen_data.lines2D(n,lib_parms)
#X = gen_data.add_outlier(20,X1,random_seed=3)

#%%
name = 'result_plots\M2R2\\'+data_name
gen_data.plotpt(X,ticks=False,savename = name,title = data_name)

#%%
n = X.shape[0]
parms={'beta':0.8,'lambda':50,\
       'list_NumPt4Instance':[2],\
       'dist_adj':True,'initial_method':'Random',}
a=RRalg(X,parms)
a.run()
name = 'result_plots\M2R2\\'+data_name +'_RRA_'+parms['initial_method']+'b08'
plot_result(a,a.list_xid_found,savename = name)
#%%
parms['initial_method'] = 'Local_Best'
a=RRalg(X,parms)
a.run()
name = 'result_plots\M2R2\\'+data_name +'_RRA_'+parms['initial_method']
plot_result(a,a.list_xid_found,savename = name)
#%%
fv, Zid = a.run_exactsol_with_C(3)
name = 'result_plots\M2R2\\'+data_name +'_exact'
plot_result(a,Zid,savename = name)
#%%