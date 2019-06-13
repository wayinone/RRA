# -*- coding: utf-8 -*-

#%%
from importlib  import reload
import gen_data
import numpy as np

import EM.fcn_fit
import EM.EM_classify as EM_classify
reload(EM.EM_classify)
reload(EM.fcn_fit)
reload(gen_data)

import RRclustering3
reload(RRclustering3)
from RRclustering3 import RRalg, save_result,load_result
import plot_tool

#%%
lib_parms={'ls_line_parm': [[0.,0.,5.,-4.],[0.,0.,1.,.5]]}
lib_parms['ls_range_x'] = [[-2,2],[-2,2]]
lib_parms['ls_var_yz'] = [.001,.001]
lib_parms['n_outlier'] = 40
lib_parms['outlier_range_x']=[-2.5,1.5]
lib_parms['outlier_range_y']=[-2.5,1.5]
n=100
X = gen_data.lines2D(n,lib_parms)
gen_data.plotpt(X,ticks=False,savename='result_plots\EM\EM_compare_original_points')
#%%
#%%
parms = {'n_clusters':2}
parms['beta_ini'] = np.array([[0.,0.,5.,-4.],[0.,0.,1.,.5]])
a = EM_classify.EM_line_model(X,parms)
a.run(tol=1e-3)
#%%
a.plot_result2D(MarkerSize=10,savename='EM_2lines_40outlier_n100')
fig.tight_layout()
#%%
parms = {'beta':0.8,'lambda':30,'list_NumPt4Instance':[2],'dist_adj':True}
b=RRclustering3.RRalg(X,parms)
b.run()
#%%
plot_tool.plot_result(b,b.list_xid_found,savename='result_plots\EM\Xshape_easy_RRa')
#%%
plot_tool.plot_result(b,b.list_xid_found,MarkerSize=50,\
                     same_color=True)
#%%
plot_tool.plot_result(b,b.list_xid_found,MarkerSize=50,\
                     same_color=True,savename='tmp')
#%%
lam_set = [5,10,50,60]
files = ['EM\RRA_result\mix_instance_b08_l%d' %lam for lam in lam_set]
#%%
plot_tool.plot_thr_result_files(files,figsize=(20,5),layout=[1,4])