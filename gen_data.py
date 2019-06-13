# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:52:56 2016

@author: Wayne
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn
from numpy import sin, cos, pi
from mpl_toolkits.mplot3d import Axes3D
import sys

#%%


def f_quadratic(x,center,amp):
    x0 = center[0]
    y0 = center[1]
    return amp*(x-x0)**2+y0
def f_cubic(C,x):
    return C[0]+C[1]*x+C[2]*x**2+C[3]*x**3

nB=30
#%%
def Scurve_hvytail(n,plot=False,dof=10):
    np.random.seed(2)
#    C = np.array([10,1,-0.005,-.02])
    x0 = 3*np.pi*np.random.rand(n,1)-.5*np.pi
    x0 = np.sort(x0)
    ep0 = .2*np.random.standard_t(dof, size=(n,1))
    ep1 = .2*np.random.standard_t(dof, size=(n,1))
    x1 = 1.5*np.sin(x0)
    X =  np.hstack((x0+ep0,x1+ep1))
    Xp = np.hstack((x0,x1))
    if plot:
        plotpt(X,'S curve heavy tail')
    return X
    
    

def Scurve(n,plot=False):
    np.random.seed(2)
    C = np.array([10,1,-0.005,-.02])    
    x0 = 20*np.random.rand(n,1)-10
    x0 = np.sort(x0)
    y0= f_cubic(C,x0)+np.random.normal(0,1,(n,1))

    xp = np.arange(0,10,.02)
    xp = np.reshape(xp,(xp.size,1))
    yp = f_cubic(C,xp)
    
    X = np.hstack((x0,y0))
    Xp = np.hstack((xp,yp))
    if plot:
        plotpt(X,'S curve')
    return X, Xp
    
def circle(n,plot=False):
    np.random.seed(2)
    e1 = np.random.randn(n,1);
    e2 = np.random.randn(n,1);
    
    lam = np.random.rand(n,1)*2*pi;
    x = 15*sin(lam)+e1;
    y = 15*cos(lam)+e2;
    
    lamp = np.arange(0,2*pi,1e-2)
    
    xp = 15*sin(lamp);
    yp = 15*cos(lamp);
    
    X = np.hstack((x,y))
    Xp = np.hstack((xp,yp))
    if plot:
        plotpt(X,'circle')
    return X,Xp

def gen_cross(n,seed=2):
    np.random.seed(seed)
    nhorz = np.ceil(n/2)
    nvert = n-nhorz
    
    xhorz = 20*rand(nhorz,1)-10
    yhorz = np.zeros((nhorz,1))
    
    xvert = np.zeros((nvert,1))
#    yvert = 20*rand(nvert,1)-10
    yvert = 5*rand(nvert,1)

    xp = np.vstack((xhorz,xvert))
    yp = np.vstack((yhorz,yvert))
    
    x = xp+.2*randn(n,1)
    y = yp+.2*randn(n,1)

    X = np.hstack((x,y))
    return X


def ellipse(n,plot=False):
    np.random.seed(2)
    e1 = randn(n,1);
    e2 = randn(n,1);
    
    lam = rand(n,1)*2*pi;
    x = 20*cos(lam)+e1;
    y = 5*sin(lam)+e2;
    
    nn=500.0
    angle = 30
    lamp = np.arange(0,2*pi,2*pi/nn)
    
    xp = 20*cos(lamp);
    yp = 5*sin(lamp);
    
    X = np.hstack((x,y))
    Xp = np.hstack((xp[...,None],yp[...,None]))
    X  = rotation(X,angle) 
    Xp = rotation(Xp,angle)
    if plot:
        plotpt(X,'ellipse')  
    return X,Xp

def helix3D(n,plot=False,n_HalfCircle=5,height=1,sd=2):
    np.random.seed(2)
    th = n_HalfCircle*pi*rand(n,1)
    x = 10*sin(th)+sd*randn(n,1)
    y = 10*cos(th)+sd*randn(n,1)
    z = height*th + sd*rand(n,1)
    X = np.hstack((x,y,z))
    if plot:
       plotpt(X,'3D helix') 
    return X

def Separated_curve(n,c1,a1,c2,a2,plot=False):
    """
    closer separate curve setting
    n=100
    c1 = np.array([-1,2])
    a1 = 0.2  
    c2 = np.array([3,-2])
    a2 = -0.1  
    """
    n1=n/2
    n2=n-n1
    np.random.seed(2)
    
    x1 = np.sort(10.0*np.random.rand(n1,1)-5+c1[0])
#    c1 = np.array([-3,3])
#    a1 = 0.4    
    y1= f_quadratic(x1,c1,a1)+np.random.normal(0,1,(n1,1))
    
    x2 = np.sort(10.0*np.random.rand(n1,1)-5+c2[0])
#    c2 = np.array([3,-3])
#    a2 = -0.5  
    y2= f_quadratic(x2,c2,a2)+np.random.normal(0,1,(n2,1))
    
    xx = np.vstack((x1,x2))    
    yy = np.vstack((y1,y2))    
    X = np.hstack((xx,yy))
    if plot:
        plotpt(X,'Separated_curve')
    return X

def planes3D(n,lib_parms):    
    """
    lib_parms['ls_plane_parm']: list of plane parms. e.g.[[1,2,3,4],[5,6,7,8]]
            means x+2y+3z+4=0 and 5x+6y+7z+8=0
    lib_parms['ls_range_xy']: list of x and y range in each plane. e.g.
            [[-5,5,-5,5],[-3,3,-3,3]] means -5<x<5, -5<y<5 for the 1st plane.
    lib_parms['ls_var_z']: list of z's fluctuate range. e.g.[3,2]
            means the z value in 1st plane is Normal(.,3)
    """
    np.random.seed(2)
    ls_plane_parm = lib_parms['ls_plane_parm']
    ls_range_xy = lib_parms['ls_range_xy']
    ls_var_z = lib_parms['ls_var_z']
    m = len(ls_plane_parm)
    ni = np.ceil(n/m)
    list_dp = [None]*m
    for i in range(m):
        if i==m-1:
            n_i=n-(m-1)*ni
        else:
            n_i = ni
        range_i = ls_range_xy[i]
        a,b,c,d = ls_plane_parm[i]
        var = ls_var_z[i]
        xr = range_i[0:2]
        yr = range_i[2:]
        xi = np.random.uniform(xr[0],xr[1],n_i)
        yi = np.random.uniform(yr[0],yr[1],n_i)
        zi = (-a*xi-b*yi-d)/c+np.random.randn(n_i)*np.sqrt(var)
        list_dp[i]=np.vstack((xi,yi,zi)).T
    X = np.vstack(list_dp)
    return X

def points(n,lib_parms):
    """
    lib_parms['ls_pts']: list of the center points. e.g.[[1,2,3],[4,5,6]]
                 'var' : list of variances. e.g.[0.3,0.2]
        (optional)
                 'ls_n': list of propotion of number of points, e.g.[0.2,0.8]
                                                                 (same as [1,4])                         
                 'distr':distrubution e.g.'Gaussian' or'Uniform
                         default:'Gaussian'
                                                                            
    
    """
    np.random.seed(2)
    list_pts = lib_parms['ls_pts']
    m = len(list_pts)
    d = len(list_pts[0])
    var = lib_parms['var']
    if 'ls_n' in lib_parms:
        ls_n = np.array(lib_parms['ls_n'],dtype='float')
        s_n = np.sum(ls_n)
        ni = [np.ceil(ls_n[i]/s_n*n) for i in range(m)]
    else:                  
        ni = [np.ceil(n/m)]*m
              
    if 'distr' in lib_parms:
        distr = lib_parms['distr']
        if distr not in ['Gaussian','Uniform']:
            print('parms[\'distr\'] can only be \'Gaussian\' or \'Uniform\'!')    
            print('Distribution is set to \'Gaussian\'.')
            distr = 'Gaussian'
        if d==3:
            print('Distribution is set to \'Gaussian\'.')
            distr = 'Gaussian'
    else:
        distr = 'Gaussian'
        
    list_dp = [None]*m

    for i in range(m):

        if i==m-1:
            n_i=n-sum(ni[:i])
        else:
            n_i = ni[i]
        if m>1:
            n_i = n_i.astype('int')
        pt_i = list_pts[i]
        std_i = np.sqrt(var[i])
        if len(pt_i)!=d:
            sys.exit('Points dimension not consistent!')
        else:
            x_i = np.zeros((n_i,d))
            if distr is 'Gaussian':
                for dd in range(d):                
                    x_i[:,dd] = pt_i[dd]+std_i*np.random.randn(n_i)
            if distr is 'Uniform':
                if d==2:
                    rr = np.random.rand(n_i,2)
                    rs = np.sort(rr,axis=1)
                    b = rs[:,1]
                    a = rs[:,0]
                    
                    
                    c=np.vstack((b*std_i*cos(2*np.pi*a/b), b*std_i*sin(2*np.pi*a/b))).T

                    x_i = pt_i+c
            list_dp[i]=x_i
    X = np.vstack(list_dp)
    return X
    
def lines2D(n,lib_parms):
    """
    lib_parms['ls_line_parm']: list of line parms. e.g.[[1,2,4,5],[5,6,8,9]]
            means there are two lines:
                1st line start with [1,2] with vector [4,5]
                2nd line start with [5,6] with vector [8,9]
    lib_parms['ls_range_x']: list of x range in each line. e.g.
            [[-5,5],[-3,3]] means -5<x<5, for the 1st plane.
    lib_parms['ls_var_y']: list of y's fluctuate range. e.g.0.3
            means the y and z value in 1st plane is Normal(.,3)
    lib_parms['n_outlier']
    """
    np.random.seed(2)
    ls_line_parm = lib_parms['ls_line_parm']
    ls_range_x = lib_parms['ls_range_x']
    ls_var_y = lib_parms['ls_var_yz']
    

    if 'n_outlier' in lib_parms:        
        n_outlier= lib_parms['n_outlier']
        xm,xM = lib_parms['outlier_range_x']
        ym,yM = lib_parms['outlier_range_y']
        xout = xm+(xM-xm)*np.random.rand(n_outlier)
        yout = ym+(yM-ym)*np.random.rand(n_outlier)    
        n = n-n_outlier
    m = len(ls_line_parm)
    ni = np.ceil(n/m)
    list_dp = [None]*m
    for i in range(m):
        if i==m-1:
            n_i=n-(m-1)*ni
        else:
            n_i = ni
        xm,xM = ls_range_x[i]
        x0,y0,rx,ry = ls_line_parm[i]

        var = ls_var_y[i]
        r = np.array([rx,ry])/np.sqrt(rx**2+ry**2)
        tm = (xm-x0)/r[0]
        tM = (xM-x0)/r[0]
        tt = tm+(tM-tm)*np.random.rand(n_i)
        
        Xp = np.array([x0,y0]+tt[:,None]*r)
        yy = Xp[:,1]+np.random.randn(n_i)*np.sqrt(var)
        
        list_dp[i]=np.vstack((Xp[:,0],yy)).T
    if 'n_outlier' in lib_parms: 
        list_dp.append(np.vstack((xout,yout)).T)
    X = np.vstack(list_dp)
    return X
def lines3D(n,lib_parms):    
    """
    lib_parms['ls_line_parm']: list of line parms. e.g.[[1,2,3,4,5,6],[5,6,7,8,9,10]]
            means there are two lines:
                1st line start with [1,2,3] with vector [4,5,6]
                2nd line start with [5,6,7] with vector [8,9,10]
    lib_parms['ls_range_x']: list of x range in each line. e.g.
            [[-5,5],[-3,3]] means -5<x<5, for the 1st plane.
    lib_parms['ls_var_yz']: list of y and z's fluctuate range. e.g.[0.3,0.3]
            means the y and z value in 1st plane is Normal(.,3)
    """
    np.random.seed(2)
    ls_line_parm = lib_parms['ls_line_parm']
    ls_range_x = lib_parms['ls_range_x']
    ls_var_yz = lib_parms['ls_var_yz']
    m = len(ls_line_parm)
    ni = np.ceil(n/m)
    list_dp = [None]*m
    for i in range(m):
        if i==m-1:
            n_i=n-(m-1)*ni
        else:
            n_i = ni
        xm,xM = ls_range_x[i]
        x0,y0,z0,rx,ry,rz = ls_line_parm[i]

        var = ls_var_yz[i]
        r = np.array([rx,ry,rz])/np.sqrt(rx**2+ry**2+rz**2)
        tm = (xm-x0)/r[0]
        tM = (xM-x0)/r[0]
        tt = tm+(tM-tm)*np.random.rand(n_i)
        
        Xp = np.array([x0,y0,z0]+tt[:,None]*r)
        yy = Xp[:,1]+np.random.randn(n_i)*np.sqrt(var)
        zz = Xp[:,2]+np.random.randn(n_i)*np.sqrt(var)
        
        list_dp[i]=np.vstack((Xp[:,0],yy,zz)).T
    X = np.vstack(list_dp)
    return X
def add_outlier(n,X_md,random_seed=2):
    np.random.seed(random_seed)
    m,d = X_md.shape
    xm = np.min(X_md,axis=0)
    xM = np.max(X_md,axis=0)
    r = xM-xm
    outliers = xm+r*np.random.rand(n,d)
    return np.vstack((X_md,outliers))
def rotation(X,angle):
    th = float(angle)/180*np.pi
    R = np.array([[cos(th),-sin(th)],[sin(th),cos(th)]])
    Xr = np.dot(R,X.T)
    return Xr.T

#def plotpt(X,ticks=True,title=None,MarkerSize=50,savename=None,elev=10.0, azim=100.0):
#    d = X.shape[1]    
#    x =X[:,0]
#    y =X[:,1]
#    xm = np.min(x);xM = np.max(x)
#    ym = np.min(y);yM = np.max(y)
#    fig = plt.figure(num=None, figsize=[5,5], dpi=80)
#    
#    if d==2:
#        ax = fig.add_subplot(111)
#        ax.scatter(x,y,marker='.',s = MarkerSize)       
##        ax.axis('equal')
#        if title is not None:
#            plt.title(title)
#        else:
#            plt.title('Original data')
#        ax.set_xlim(xm,xM)
#        ax.set_ylim(ym,yM)
#        if ticks==0:
#            ax.set_xticks([],[])
#            ax.set_yticks([],[])
#    if d==3:
#        z = X[:,2]
#        cmap = plt.get_cmap('jet')
#        ax = fig.add_subplot(111, projection='3d')
#        
#        zm = np.min(z);zM = np.max(z)
#        ax.scatter(x,y,z,s=MarkerSize,marker='.',linewidth='0.5')
##        ax.scatter(x,y,z,c=z,s = Markersize, cmap=cmap)
#        ax.auto_scale_xyz([xm,xM], [ym, yM], [zm, zM])
#        ax.view_init(elev=elev, azim=azim)
#        if ticks==0:
#            ax.set_zticks([],[])
#        if title is not None:
#            fig.suptitle(title)
#        else:
#            fig.suptitle('Original data')
##        plt.zlim([zm-0.2,zM+0.2]) 
#  
#    if savename is not None:
#        fig.tight_layout()
#        plt.savefig(savename+'.png')
def plotpt(data,ticks=True,ax = None, MarkerSize=50,\
                elev=10.0, azim=100.0,\
                title = None,savename = None):
    """
    title=None: suppress the title
    list_zeta: list of zeta parameter, list_zeta[p]=zeta_p, p=1,2,3,...
    """
    
    n,d = data.shape

    # 1. plot original data point
    X0 = data[:,0];Y0 = data[:,1];
    xm = np.min(X0);xM = np.max(X0)
    ym = np.min(Y0);yM = np.max(Y0)    
    if d==2:
        
        if ax is None:
            fig = plt.figure(num=None, figsize=[5,5], dpi=80)
            ax = fig.add_subplot(111)
        ax.scatter(X0,Y0,s=MarkerSize,marker='.',linewidth='0.5')
    if d==3:
        Z0 = data[:,2]
        zm = np.min(Z0);zM = np.max(Z0)
        if ax is None:
            fig = plt.figure(num=None, figsize=[5,5], dpi=80)
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X0,Y0,Z0,c=Z0,s=MarkerSize,marker='.',linewidth='0.5',cmap = plt.get_cmap('jet'))        
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
        ax.set_title(title)
    else:
        ax.set_title('original data')
    
    
    if savename is not None:
        fig.tight_layout()
        plt.savefig(savename+'.png')
    return ax
