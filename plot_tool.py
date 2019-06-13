# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 04:57:13 2017

@author: Wayne
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, colors
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import warnings
#%%
def plot_result(a,list_xid_found, list_zeta = 'default',ax = None, MarkerSize=50,\
                elev=10.0, azim=100.0,\
                title = 'default',title_add = None,savename = None,same_color=False):
    """
    title=None: suppress the title
    list_zeta: list of zeta parameter, list_zeta[p]=zeta_p, p=1,2,3,...
    """
    data= a.data
    n,d = data.shape
    colorname = list(colors.cnames.keys())
    colorname.pop(2)
    colorname = ['r','g','b','c','m','y','k']+colorname
    if list_zeta is 'default':
        list_zeta =  a.list_adj_quotient
    try:    
        cc,sum_err = a.fcn_classify(list_xid_found,list_zeta)
    except:
        warnings.warn('list of zeta out of range, change zeta =1')
        list_zeta = [1]*10
        cc,_ = a.fcn_classify(list_xid_found,list_zeta)
    if same_color:
        clist = ['b' for k in range(n)]
    else:
        clist = [colorname[cc[k]] for k in range(n)]

    m = len(list_xid_found)
    _,data_var = standerdizing(data)
    adj = np.sqrt(data_var)/8
    
    # 1. plot original data point
    X0 = data[:,0];Y0 = data[:,1];
    xm = np.min(X0);xM = np.max(X0)
    ym = np.min(Y0);yM = np.max(Y0)    
    if d==2:
        
        if ax is None:
            fig = plt.figure(num=None, figsize=[5,5], dpi=80)
            ax = fig.add_subplot(111)
        ax.scatter(X0,Y0,s=MarkerSize,c=clist,marker='.',linewidth='0.5')
    if d==3:
        Z0 = data[:,2]
        zm = np.min(Z0);zM = np.max(Z0)
        if ax is None:
            fig = plt.figure(num=None, figsize=[5,5], dpi=80)
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X0,Y0,Z0,s=MarkerSize,c=clist,marker='.',linewidth='0.5')
    # 2. plot each instances    
    for i in range(m):
        data_i = data[cc==i]
        color_i = colorname[i]
        Z_ids = list_xid_found[i]# Z_ids is (p,) array, array of indexes of data
        Z = data[Z_ids,:]# (p,d)    
        if d==2:
            plot_2D_structure(Z,data_i,ax,color_i,adj, MarkerSize)
        if d==3:           
            plot_3D_structure(Z,data_i,ax,color_i,adj, MarkerSize)
        
    # 3. set plot parameters
    ax.set_xticks([], []);ax.set_yticks([], []);
    if d==2:
        ax.set_xlim(xm,xM)
        ax.set_ylim(ym,yM)
    if d==3:
        ax.auto_scale_xyz([xm,xM], [ym, yM], [zm, zM])
        ax.view_init(elev=elev, azim=azim)
        ax.set_zticks([], [])
    
    if title is not None:
        if title is 'default':
            title=r'$\beta=%1.1f,\lambda=%1.0f,h=%1.3f$' %(a.beta,a.lam,sum_err)
        if title_add is not None:
            title1=r'$,\beta=%1.1f,\lambda=%1.0f$,' %(a.beta,a.lam)
            title = title_add+title1
        if title is 'exact':
            title = r'Exact Sol. $\beta=%1.1f,h=%1.3f$' %(a.beta,sum_err)
        
        if title is 'show_weight':
            title1=r'$\beta=%1.1f,\lambda=%1.0f$,' %(a.beta,a.lam)
            title2='['
            title3 = '['
            for j in range(len(a.omega)):
                j1 = j+1
                title2+=r'$\omega_{%d}$,' %j1
                title3+=r'$%1.1f$,' %a.omega[j]

            title = title1+title2[:-1]+']='+title3[0:-1]+']'
        ax.set_title(title)
    if savename is not None:
        fig.tight_layout()
        plt.savefig(savename+'.png')
    return ax

def plot_2D_structure(Z,data_i,ax,color_i,adj=1,ax_label=None,MarkerSize=50):
    """
    Plot one instance at a time.
    Z: (p,d),  a data point
    """
    p = Z.shape[0]
    if p==1:
        x,y = Z.flatten()
        ax.scatter(x,y,c=color_i,marker='o',label = ax_label,s = MarkerSize*2)
    if p==2:
        X0 = Z[0,:]
        vec = Z[1,:]-X0
        r  = vec/np.sqrt(np.sum(vec**2))
        j = np.argmax(np.abs(r))
        tm = (np.min(data_i[:,j])-X0[j]-adj)/r[j]
        tM = (np.max(data_i[:,j])-X0[j]+adj)/r[j]        
        line_c = np.array([X0+t*r for t in [tm,tM]])
        ax.plot(line_c[:,0],line_c[:,1],c = color_i,label = ax_label)
        
def plot_3D_structure(Z,data_i,ax,color_i,adj=1,ax_label=None,MarkerSize=50):
    """
    Plot one instance at a time.
    Z: (3,), a data point
    """
    p = Z.shape[0]
    if p==1:
        x,y,z = Z.flatten()
        ax.scatter(x,y,z,c=color_i,marker='o',label = ax_label,s = MarkerSize*2)
    if p==2:
        X0 = Z[0,:]
        vec = Z[1,:]-X0
        r  = vec/np.sqrt(np.sum(vec**2))
        j = np.argmax(np.abs(r))
        tm = (np.min(data_i[:,j])-X0[j]-adj)/r[j]
        tM = (np.max(data_i[:,j])-X0[j]+adj)/r[j]        
        line_c = np.array([X0+t*r for t in [tm,tM]])
        ax.plot(line_c[:,0],line_c[:,1],line_c[:,2],c = color_i,label = ax_label)
    if p==3:
        pt = Z[0,:][None,:]        
        r1 = Z[1,:]-pt# size (n-i-1,3)
        r2 = Z[2,:]-pt# size (n-i-2,3)
        nvec = np.cross(r1,r2)
        tmp = np.sqrt(np.sum(nvec**2)) # normalize           
        normal=(nvec.T/tmp)
        
        cnst = -np.dot(pt,normal)
        
        X = data_i[:,0];Y = data_i[:,1];Z = data_i[:,2]
        xm = np.min(X);xM = np.max(X)
        ym = np.min(Y);yM = np.max(Y)
        zm = np.min(Z);zM = np.max(Z)
        
        xr = np.arange(xm,xM,.1)
        yr = np.arange(ym,yM,.1)
        zr = np.arange(zm,zM,.1)
        if np.abs(normal[2])>1e-1:
            xx, yy = np.meshgrid(xr,yr)
            zz = (-normal[0] * xx - normal[1] * yy - cnst)/normal[2]
        elif np.abs(normal[1])>1e-1: 
            xx,zz = np.meshgrid(xr,zr)
            yy = (-normal[0] * xx - normal[2] * zz - cnst)/normal[1]
        else:
            yy,zz= np.meshgrid(yr,zr)
            xx = (-normal[1] * yy - normal[2] * zz - cnst)/normal[0]
        ax.plot_surface(xx,yy,zz,color=color_i,alpha=0.3,  shade=True,
                        linewidth=0,label = ax_label)
        
def plot_thr_result_files(list_file,MarkerSize=50,figsize=(8, 6),layout='None',\
                          savename='tmp',same_color=False,elev=10.0, azim=100.0):
    from RRclustering3 import load_result
    fig = plt.figure(num=None, figsize=figsize, dpi=80)
    
    K = len(list_file)
    i = 1
    if layout is None:        
        K1 = np.ceil(np.sqrt(K))
        K2 = np.floor(np.sqrt(K))
    else:
        K1 = layout[1]
        K2 = layout[0]
    if K1*K2<K:
        K2 = K2+1
    for f in list_file:
        g = load_result(f)
        d = g.data.shape[1]
        if d==2:
            axe = fig.add_subplot(K2,K1,i)
            plot_result(g,g.list_xid_found,ax = axe,MarkerSize = MarkerSize,same_color=same_color)
        if d==3:
            axe = fig.add_subplot(K2,K1,i,projection='3d')
            plot_result(g,g.list_xid_found,ax = axe,MarkerSize = MarkerSize,\
                        same_color=same_color,elev=elev, azim=azim)
        i +=1
    fig.tight_layout()
    plt.savefig(savename+'.png')

def norm(Xnd):
    c = np.sqrt(np.sum(Xnd**2,axis=1))
    return c
def standerdizing(Y_nd):
    mean = np.mean(Y_nd,axis=0)
    tmp=np.var(Y_nd,axis=0)
    data_var = np.sum(tmp)
    X_nd=(Y_nd-mean)/np.sqrt(data_var)
    return X_nd,data_var