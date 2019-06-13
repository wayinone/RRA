# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:08:29 2017

@author: wayin
"""

#%%
import os
import numpy as np
import ctypes
#%%

#dllabspath = os.path.dirname(os.path.abspath('__file__')) + \
#            os.path.sep +\
#            'RRA'+ os.path.sep 
dllabspath = os.path.dirname(os.path.abspath('__file__')) + \
            os.path.sep +\
             os.path.sep 
#my_lib1 = ctypes.CDLL('./data_term_addz.so')
my_lib1 = ctypes.CDLL(dllabspath+ 'data_term_addz.so')
data_term_addz_ = my_lib1.data_term_addz
def data_term_addz_C(A_nK,arrmin_xz_n):
    """
    A_nK: float32 (n,K) numpy array
    """
    A_nK = A_nK.astype('float32')
    arrmin_xz_n = arrmin_xz_n.astype('float32')
    n = A_nK.shape[0]
    K = A_nK.shape[1]
    
    y = np.zeros((K,),dtype='float32')
    
    AA = A_nK.copy()
    b = arrmin_xz_n.copy()
    data_term_addz_(AA.ctypes.data_as(ctypes.c_void_p),b.ctypes.data_as(ctypes.c_void_p),\
                    n,K,y.ctypes.data_as(ctypes.c_void_p)) 
    return y
##%%
#A_nN = np.array([[1,7,4,0,9],[5,1,7,1,1],[2,3,2,2,1],[8,9,2,7,9.]])
#arrmin_xz_n = np.array([5.,4,1,3])
##%%
#y=data_term_addz_C(A_nN,arrmin_xz_n)
##%%
#A_nN = a.dist_xz[:,0:35]
#arrmin_xz = np.append(0.054,np.zeros(34))
#
#y=data_term_addz_C(A_nN,arrmin_xz)
#%%
my_lib2 = ctypes.CDLL(dllabspath+ 'exactsol_m2.so')
exactsol_m2_ = my_lib2.exactsol_m2
def exactsol_m2(A_nN):
    """
    A_nN: float32 (n,N) numpy array
    """
    A_nN = A_nN.astype('float32')
    n = A_nN.shape[0]
    N = A_nN.shape[1]
    Zid = np.zeros((2,),dtype='int32')
    AA = A_nN.copy()
    exactsol_m2_(AA.ctypes.data_as(ctypes.c_void_p),n,N,	Zid.ctypes.data_as(ctypes.c_void_p)) 
    return Zid
#%%    
my_lib3 = ctypes.CDLL(dllabspath+ 'm3_exactsol.so')
exactsol_m3_ = my_lib3.exactsol_m3

def exactsol_m3(A_nN):
    """
    A_nN: float32 (n,N) numpy array
    """
    A = A_nN.astype('float32')    
    n = A.shape[0]
    N = A.shape[1]
    Zid = np.zeros((3,),dtype='int32')
    AA = A.copy()
    exactsol_m3_(AA.ctypes.data_as(ctypes.c_void_p),n,N,	Zid.ctypes.data_as(ctypes.c_void_p)) 
    return Zid