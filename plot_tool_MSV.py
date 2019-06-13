# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:30:44 2017

@author: wayin
"""

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
import mpl_toolkits.mplot3d.art3d as art3d
import warnings
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
#%%
def plot_result(b,list_xid_found,plot_volume=True,ax = None, MarkerSize=50,\
                elev=10.0, azim=100.0,\
                title = 'default',title_add = None,\
                ticks = False,savename = None,same_color=False):
    """
    title=None: suppress the title
    list_zeta: list of zeta parameter, list_zeta[p]=zeta_p, p=1,2,3,...
    """
    data= b.X_nd
    n,d = data.shape
    colorname = list(colors.cnames.keys())
    colorname.pop(2)
    colorname = ['r','g','b','c','m','y','k']+colorname
    
    cc = b.fcn_classify(list_xid_found)

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
        if plot_volume:
            r_in   = b.dict_msv['r_in_msv'][i]
            r_dist = b.dict_msv['r_dist_msv'][i]
            mu_in  = b.dict_msv['mu_in_msv'][i,:]
            if d==2:
                plot_2D_MSV_structure(Z,data_i,r_in,r_dist,mu_in,ax,color_i,adj, MarkerSize)
            if d==3:           
                plot_3D_MSV_structure(Z,data_i,r_in,r_dist,mu_in,ax,color_i,adj, MarkerSize)
        else:
            if d==2:
                plot_2D_structure(Z,data_i,ax,color_i,adj, MarkerSize)
            if d==3:           
                plot_3D_structure(Z,data_i,ax,color_i,adj, MarkerSize)
        
    # 3. set plot parameters
    if ticks==0:
        ax.set_xticks([], []);
        ax.set_yticks([], []);

    if d==2:
        ax.set_xlim(xm,xM)
        ax.set_ylim(ym,yM)
        
    if d==3:
        ax.auto_scale_xyz([xm,xM], [ym, yM], [zm, zM/5])
        ax.set_xlim(xm,xM)
        ax.set_ylim(ym,yM)
        ax.set_zlim(zm,zM)
        ax.view_init(elev=elev, azim=azim)
        if ticks==0:
            ax.set_zticks([], [])
    
    if title is not None:
        if title is 'default':
            title=r'$\beta=%1.1f,\lambda=%1.0f, MSV=%1.1f$' %(b.beta,b.parms['lambda'], b.dict_msv['MSV'])
        if title_add is not None:
            title1=r'$\beta=%1.1f,\lambda=%1.0f, MSV=%1.1f$' %(b.beta,b.parms['lambda'], b.dict_msv['MSV'])
            title =title1+ ','+title_add
        
        ax.set_title(title)
    
    if savename is not None:
        fig.tight_layout()
        plt.savefig(savename+'.png')
    return ax

def plot_2D_MSV_structure(Z,data_i,r_in,r_dist,mu_in,ax,color_i,adj=1,ax_label=None,MarkerSize=50):
    """
    Plot one instance at a time.
    Z: (p,d),  a data point that instance pass through
    """
    p = Z.shape[0]
    if p==1:
        x,y = Z.flatten()
        ax.scatter(x,y,c=color_i,marker='o',label = ax_label,s = MarkerSize*2)
        th = np.arange(-0.2,2*np.pi,1e-1)
        xv = r_dist*np.cos(th)+x
        yv = r_dist*np.sin(th)+y
        ax.plot(xv,yv,c=color_i,alpha=0.3)
    if p==2:
        X0 = Z[0,:]
        vec = Z[1,:]-X0
        r  = vec/np.sqrt(np.sum(vec**2))
        j = np.argmax(np.abs(r))
        tm = (np.min(data_i[:,j])-X0[j]-adj)/r[j]
        tM = (np.max(data_i[:,j])-X0[j]+adj)/r[j]
        line_c = np.array([X0+t*r for t in [tm,tM]])
        ax.plot(line_c[:,0],line_c[:,1],c = color_i,label = ax_label)    

        r_per = r.copy()
        if r[0]>r[1]:
            r_per[0]=r[1];r_per[1]=-r[0]
        else:
            r_per[0]=-r[1];r_per[1]=r[0]
        Xcen = mu_in
#        ax.scatter(mu_in[0],mu_in[1],c=color_i,marker='o',label = ax_label,s = MarkerSize*2)
        X1 = Xcen+r_dist*r_per
        X2 = Xcen-r_dist*r_per
        pts = np.vstack([X1+r_in*r,X1-r_in*r,X2-r_in*r,X2+r_in*r,X1+r_in*r])
        ax.plot(pts[:,0],pts[:,1],c = color_i,label = ax_label,alpha=0.3)    

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
def plot_3D_MSV_structure(Z,data_i,r_in,r_dist,mu_in,ax,color_i,adj=1,ax_label=None,MarkerSize=50):
    """
    Plot one instance at a time.
    Z: (3,), a data point
    """
    p = Z.shape[0]
    if p==1:
        #%% center point
        x,y,z = Z.flatten()
        ax.scatter(x,y,z,c=color_i,marker='o',label = ax_label,s = MarkerSize*2)
        #%% balls in R3
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)  
        xx = x+r_dist * np.outer(np.cos(u), np.sin(v))
        yy = y+r_dist * np.outer(np.sin(u), np.sin(v))
        zz = z+r_dist * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xx, yy, zz,  color=color_i, linewidth=0, alpha=0.2)
    if p==2:
        X0 = Z[0,:]
        vec = Z[1,:]-X0
        r  = vec/np.sqrt(np.sum(vec**2))
        #%% line instance
        j = np.argmax(np.abs(r))
        tm = (np.min(data_i[:,j])-X0[j]-adj)/r[j]
        tM = (np.max(data_i[:,j])-X0[j]+adj)/r[j]        
        line_c = np.array([X0+t*r for t in [tm,tM]])
        ax.plot(line_c[:,0],line_c[:,1],line_c[:,2],c = color_i,label = ax_label)
        #%%cylinder
        r_per = r.copy()
        if r[0]>r[1]:
            r_per[0]=r[1];r_per[1]=-r[0];
        else:
            r_per[0]=-r[1];r_per[1]=r[0]

        R = vec2rotMat3D(r)
        
        Xcen = mu_in
        Xcen1 = Xcen-r_in*r
        Xcen2 = Xcen+r_in*r
        
        k=100
        th = np.linspace(0,2*np.pi,k)
        circle_xy = r_dist*np.vstack((np.cos(th),\
                                      np.sin(th),\
                                      np.zeros(len(th))))
        xx1 = Xcen1+np.dot(R,circle_xy).T
        xx2 = Xcen2+np.dot(R,circle_xy).T
        xx3 = np.zeros((2*k,3))
        for j in range(k):
            xx3[2*j  ,:]=xx1[j  ,:]
            xx3[2*j+1,:]=xx2[j,:]
        ax.plot(xx3[:,0],xx3[:,1],xx3[:,2],color = color_i,linewidth=1, alpha=0.2)
        ax.plot(xx1[:,0],xx1[:,1],xx1[:,2],color = color_i,linewidth=1, alpha=0.2)
        ax.plot(xx2[:,0],xx2[:,1],xx2[:,2],color = color_i,linewidth=1, alpha=0.3)
#        xxl = np.vstack((Xcen1[0]+r_dist*np.cos(th),\
#                         Xcen1[1]+r_dist*np.sin(th),\
#                         Xcen1[2]+np.zeros(len(th))))
#        xxl = np.dot(R,xxl).T
#        ax.plot_surface(xxl[:,0],xxl[:,1],xxl[:,2],linewidth=0, alpha=0.2)
#        
#        xxr = np.vstack((Xcen2[0]+r_dist*np.cos(th),\
#                 Xcen2[1]+r_dist*np.sin(th),\
#                 Xcen2[2]+np.zeros(len(th))))
#        xxr = np.dot(R,xxr).T
#        ax.plot_surface(xxr[:,0],xxr[:,1],xxr[:,2],linewidth=0, alpha=0.2)
    if p==3:
        pt = Z[0,:][None,:]        
        r1 = Z[1,:]-pt# size (n-i-1,3)
        r2 = Z[2,:]-pt# size (n-i-2,3)
        nvec = np.cross(r1,r2)
        tmp = np.sqrt(np.sum(nvec**2)) # normalize           
        normal=(nvec.T/tmp)
        
        cnst = -np.dot(pt,normal)
        #%% surface in R3
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
        ax.plot_surface(xx,yy,zz,color=color_i,alpha=0.5,  shade=True,
                        linewidth=0,label = ax_label)
        
        #%% disk in R3
#        k=1000
        xx, yy = np.meshgrid(range(10), range(10))
        k=100
        th = np.linspace(0,2*np.pi,k)
        circle_xy = r_in*np.vstack((np.cos(th),\
                                      np.sin(th),\
                                      np.zeros(len(th))))
        Xcen = mu_in
        normal = normal.flatten()

        Xcen1 = Xcen+normal*r_dist
        Xcen2 = Xcen-normal*r_dist
        R = vec2rotMat3D(normal)
  
        xx1 = Xcen1+np.dot(R,circle_xy).T
        xx2 = Xcen2+np.dot(R,circle_xy).T
        ax.plot(xx1[:,0],xx1[:,1],xx1[:,2],color=color_i,alpha=0.2,linewidth=1)
        ax.plot(xx2[:,0],xx2[:,1],xx2[:,2],color=color_i,alpha=0.2,linewidth=1)
        #%%
        Xcen = mu_in
        Xcen1 = Xcen+normal*r_dist
        Xcen2 = Xcen-normal*r_dist
        
        xx,yy = circle_mesh(Xcen,r_in)
        zz = np.zeros_like(xx)
        circle_xy = np.vstack((xx,yy,zz))
        
        R = vec2rotMat3D(normal)  
        xx1 = Xcen1+np.dot(R,circle_xy).T
        xx2 = Xcen2+np.dot(R,circle_xy).T

        ax.plot(xx1[:,0],xx1[:,1],xx1[:,2],color=color_i,alpha=0.2,linewidth=1)
        ax.plot(xx2[:,0],xx2[:,1],xx2[:,2],color=color_i,alpha=0.2,linewidth=1)
        
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
        

def norm(Xnd):
    c = np.sqrt(np.sum(Xnd**2,axis=1))
    return c
def standerdizing(Y_nd):
    mean = np.mean(Y_nd,axis=0)
    tmp=np.var(Y_nd,axis=0)
    data_var = np.sum(tmp)
    X_nd=(Y_nd-mean)/np.sqrt(data_var)
    return X_nd,data_var
def vec2rotMat3D(b):
    """
    b is a normalized 3D vector
    This rotate (0,0,1) to b
    """
    if b[2]>1-1e-4:
        R = np.eye(3)
    else:
        v = np.cross(np.array([0,0,1]),b)
        c = b[2]
        M = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3)+M+np.dot(M,M)/(1+c)
    return R        

def circle_mesh(center,radius,k=50):
    c = center
    r = radius
    xr = np.linspace(c[0]-r-1,c[0]+r+1,k)
    yr = np.linspace(c[1]-r-1,c[1]+r+1,k)
    xx, yy = np.meshgrid(xr,yr)
    xx = xx.flatten()
    yy = yy.flatten()
    ids = (xx**2+yy**2<=r**2)
    xx = xx[ids]
    yy = yy[ids]
    return xx,yy