# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:59:54 2017

@author: wayin
"""
import numpy as np
import time
import itertools as it
import dill
import sys
from wrapper_fcns import data_term_addz_C,exactsol_m2, exactsol_m3
import warnings
#%%
"""
Class:
    RRalg:
        Input: 
            data: (n,d) numpy float64 array.
            parms: see RRalg document.
            e.g.
                parms = {'beta':0.8,'lambda':50,'list_NumPt4Instance':[2],'dist_adj':True}
                a=RRclustering3.RRalg(X,parms)
        Excutable function:
            1. a.run(): The main algorithm, the most important attribute to get is:
                        a.list_xid_found: list of m instances.
                                each instance is a numpy array of int, 
                                which is consisted with p data points.
                                    If p=1: this instance is a point.
                                    If p=2: this instance is a line.
                                    If p=3: this instance is a plane.
                                    ...etc.
                        a.se: sum of residuals.
                              which is used to compare the result with exact sol.
                        a.fvals: value of the target function.      
                    
            2. a.run_exactsol_with_C()
                    Input: 
                        m: the number of instances
                        Note: Only support m=1,2,3
                        e.g. a.run_exactsol_with_C(2)
                    Output: fv,xid_found
                        fv: sum of residuals.
                        xid_found: similar to a.list_xid_found from a.run().
                        
            
Function:  
    save_result: String. Save the RRalg result. 
                e.g. save_result(a,'tmp') 
                     This will save the result as 'tmp.rra'
    load_result: load the .rra file.
                e.g. load_result('tmp')

"""

class RRalg(object):
    def __init__(self,data,parms,result=None):
        """
        Remove or replace Algorithm
        data: (n,d) numpy array
        parms: dictionary with
            'beta': a real value in (0,1)
            'lambda': a real value in (0,\infty) 
            'list_NumPt4Instance': an integer p, the number 
                    of parameters to determine a instance.
                    e.g.[1,2,3]: will examing p=1,p=2,and p=3 case
                    Note: 'plane model' is equivalent to p=3;
                          'line model' is equivalent to p=2.
                    e.g. [2]: only examing p=2 case.
            
            Optional:
                    
            'initial_method': 'Random' or 'Local_Best'
                            Default: 'Random'
            'ini_seed': an interger>0. Only valid if 'initial_method' is 'Random':
                                 set this if you want random start with different seed.
            'list_dist_mply': list of multiplier, the length of the list is the
                            same as 'list_NumPt4Instance' 
                            e.g. [1.,1.2,2.4]: multiply 1,1.2, and 2.4 on distance
                                matrix (dist_xy) from p=1,2,and 3, respectively.
                            Default: 1
            'dist_adj':True or False. adjust the distances by the mean distance
                    of its kind. (adjust the difference dist_xz of differnt p)
                    Defaule: True
            'disp_info':True or False. Display the information during 'run'
                    Default: False, this suppress the information.

        """
        self.data=data
        self.beta  = parms['beta']
        self.lam   = parms['lambda']

        self.parms=parms
        
        self.n,self.d = data.shape
        # standerdize the data:
        self.Xnd,self.data_se = standerdizing(data)
        if result is not None:
            self.list_xid_found  = result['list_xid_found']
            self.fvals   = result['fvals']
            self.Zini    = result['Zini']
            self.se      = result['SumOfError']
        if 'list_Allz_M' in parms:
            list_Allz_M = parms['list_Allz_M']
            M = len(list_Allz_M)
            self.dist_xz= self.get_dist_xz_from_z(list_Allz_M)
            b = list_Allz_M
            b_ary = np.array([len(b[i]) for i in range(M)])
            list_p = list(np.sort(np.unique(b_ary)))
            len_p = len(list_p)
            self.list_p = list_p
            self.n_diff_p = len_p
             
            N_diff_p = np.zeros(len_p,dtype = 'int')
            list_Allz_Np = [None]*(np.max(list_p)+1)
            count = 0
            for p in list_p:
                ids = np.nonzero(b_ary==p)[0]
                N_diff_p[count] = np.sum(b_ary==p)
                tmp = [list_Allz_M[ids[i]] for i in range(len(ids))]
                list_Allz_Np[p] = np.vstack(tmp)
                count+=1
            self.list_Allz_Np   = list_Allz_Np  
            self.N_diff_p = N_diff_p
            self.N = self.dist_xz.shape[1]
        if hasattr(self,'list_Allz_Np') is False:
            self.list_NumPt4Instance = parms['list_NumPt4Instance']
            list_p = parms['list_NumPt4Instance']
            self.list_p = list_p
            len_p = len(list_p)
            self.n_diff_p = len_p
            if np.max(list_p)>self.d:
                sys.exit('parms[\'list_NumPt4Instance\'] must be list of integers,p, s.t.1<p<d+1, which determine the number of data points needed for an instance.')
         
            if 'list_dist_mply' in parms:
                dist_multiplier = parms['list_dist_mply']
                self.omega = dist_multiplier
                if len(dist_multiplier)!=len_p:
                    sys.exit('Parameter \'dist_multiplier\' must have same length of \'list_NumPt4Instance\'.')
            else:
                dist_multiplier = [1]*len_p
            if 'dist_adj' in parms:
                dist_adj = parms['dist_adj']
            else:
                parms['dist_adj'] = True
                dist_adj = True
            list_dist_nN = [None]*len_p
            list_Allz_Np = [None]*(np.max(list_p)+1)
            N_diff_p   = np.zeros(len_p,dtype='int')
            list_adj_quotient = [None]*(np.max(list_p)+1)
            for l in range(len_p):
                p = list_p[l]
                d_nN0,list_Allz_Np[p],N_diff_p[l],list_adj_quotient[p] = self.get_dist_xz(p,dist_multiplier[l],dist_adj)
                list_dist_nN[l] = d_nN0
            self.dist_xz = np.hstack(list_dist_nN)
            self.list_Allz_Np = list_Allz_Np
            self.N_diff_p = N_diff_p
            self.N = self.dist_xz.shape[1]
            self.list_adj_quotient=list_adj_quotient
            
        if 'disp_info' in parms:
            self.disp_info = parms['disp_info']
        else:
            self.disp_info = False
        
        if len(parms['list_NumPt4Instance'])==1:
            if 'initial_method' in parms:
                self.ini_method = parms['initial_method']
            else:
                self.ini_method = 'Random'
        else:
            if 'initial_method' in parms:
                if parms['initial_method'] is 'Local_Best':
                    
                    warning.warn('The module has not yet support local best for\
                                 multiple models, switch to \'Random initial.\'')            
            self.ini_method = 'Random'
            
        if self.ini_method is 'Random':            
            if 'ini_seed' in parms:
                self.seed = parms['ini_seed']
            else:
                self.seed = 1
        if 'fcn_gamma' in parms:
            """
            fcn_gamma: function mapping array of p to a real value
            p[i]: number of data point used for instance i
            """
            self.gamma = parms['fcn_gamma']
        else:
            def square(x):
                return len(x)**2
            self.gamma = square

    def Zind2p_lite(self,Zind):
        N_ary = self.N_diff_p
        count = 0
        Zid_wrt_p = Zind
        while N_ary[count]<=Zid_wrt_p:
            Zid_wrt_p-=N_ary[count]
            count +=1
        return self.list_p[count],Zid_wrt_p
    def Zind2p(self,Zind_ary):
        m = len(Zind_ary)
        p = np.zeros(m,dtype='int')
        Zid_wrt_p = np.zeros(m,dtype='int')
        for i in range(m):
            p[i],Zid_wrt_p[i]=self.Zind2p_lite(Zind_ary[i])
        return p,Zid_wrt_p
    def get_dist_xz(self,p,multiplier=1,dist_adj=True):
        """ 
        
        p: p=1:point model; p=2:line model;p=3:plane model;
            p is the number of data point that determine the instance.
        N: N choose p
        Allz_Np: (N,p) array
        dist_xz: (n,N) array
        """
        Xnd = self.Xnd
        n,d = Xnd.shape
        N_max = nchoosek(n,p)
        #%% 1. get Allz_Np
        #i = 0,1,2,...,n-3
        #j = i+1,i+2,...,n-2
        #k = j+1,J=2,...,n-1
        count = 0
        Allz_Np = np.zeros((N_max,p),dtype='int')
        dist_nN = np.zeros((n,N_max))
        for i in range(n-p+1):
            Xi = Xnd[i,:]
            Yi_dn = (Xnd-Xi).T #(d,n)
            if p==1:
                di = np.sqrt(np.sum(Yi_dn**2,axis=0))**self.beta #(n,)
                dist_nN[:,count] = di
                Allz_Np[count]=i
                count+=1
            elif p>1:
                tmp = it.combinations(np.arange(i+1,n),p-1)
                AllComb = np.vstack(tmp) #(Ni,p-1)
                Ni = AllComb.shape[0]
                for x in range(Ni):
                    A = Yi_dn[:,AllComb[x]]#(d,p)
                    try:
                        AtAinv = np.linalg.inv(np.dot(A.T,A))#(2,2)
                        Allz_Np[count,:] = np.hstack((i,AllComb[x,:]))                
                        AtAinv_At = np.dot(AtAinv,A.T)
                        M_dd = np.eye(d)-np.dot(A,AtAinv_At)#(d,d)
                        proj_dn = np.dot(M_dd,Yi_dn)
                        di_jk_n = np.sqrt(np.sum(proj_dn**2,axis=0))**self.beta#(n,)
    
                        dist_nN[:,count] = di_jk_n
                        count+=1
                    except:
                        pass
        if count<N_max:
            Allz_Np = np.delete(Allz_Np,np.arange(count,N_max),axis = 0)
            dist_nN = np.delete(dist_nN,np.arange(count,N_max),axis = 1)
        N = Allz_Np.shape[0]
        
        # calculate adjust distance for each instances
        if dist_adj:
            adj = np.mean(dist_nN.flatten())
            dist_nN = dist_nN/adj*multiplier
        else:
            adj =1
            if multiplier!=1:
                dist_nN = dist_nN*multiplier
        
        return dist_nN,Allz_Np,N,adj
    def get_dist_xz_from_z(self,list_Allz_M):
        """
        list_Allz_M: list of M, each element consist p indexes of data.
        """
        M = len(list_Allz_M)
        Xnd = self.Xnd
        n,d = Xnd.shape
        dist_nN = np.zeros((n,M))
        for l in range(M):
            xid_l = list_Allz_M[l]
            xid = xid_l[0]
            Xi = Xnd[xid,:]
            Yi_dn = (Xnd-Xi).T #(d,n)
            p =len(xid_l)
            if p==1:
                di = np.sqrt(np.sum(Yi_dn**2,axis=0))**self.beta #(n,)
                dist_nN[:,l] = di
                
            elif p>1:
                A = Yi_dn[:,xid_l[1:]]#(d,p-1)
                AtAinv = np.linalg.inv(np.dot(A.T,A))#(p-1,p-1)              
                AtAinv_At = np.dot(AtAinv,A.T)
                M_dd = np.eye(d)-np.dot(A,AtAinv_At)#(d,d)
                proj_dn = np.dot(M_dd,Yi_dn)
                di_jk_n = np.sqrt(np.sum(proj_dn**2,axis=0))**self.beta#(n,)
                dist_nN[:,l] = di_jk_n
        return dist_nN
            

    def fcn_id_wrt_AllComb(self,p,ary_kp):
        """
        cannot be used when len(parms['list_NumPt4Instance'])>2, 
        i.e. cannot be used when using multiple models
        
        ary_kp: (k,d) array. 
        For each row of ary_kp, say ary_kp[k,:]=[i,j,l], this function return 
        the index of line (or planes) corresponds to line (or plane) generated 
        by data i,j,l. The index of the line gives line (or plane) in self.Allz_Np.
        """
        k,p = ary_kp.shape
        ary_kp_sort = np.sort(ary_kp,axis=1)
        AllComb = self.list_Allz_Np[p]
        N = AllComb.shape[0]
        corsZ = np.zeros((k,),dtype='int')# The corresponding Zid
        count = 0
        for nn in range(k):
            query = ary_kp_sort[nn,:]
            TFary = np.ones((N,),dtype = 'bool')
            for pp in range(p):
                tmp = AllComb[:,pp]==query[pp]
                TFary = TFary*tmp
            b = np.nonzero(TFary)[0]
            if b.size==1:                
                corsZ[count]=b
                count +=1
        if count<k:
            corsZ = np.delete(corsZ,np.arange(count,k))
        return corsZ
    def run(self):
        tic = time.time()
        disp_info = self.disp_info#
        if self.ini_method is 'Random':
            Znew = self.Zind_ini_rand(self.seed)
            print('Initial Method: Random initial.')
        elif self.ini_method is 'Local_Best':
            Znew = self.Zind_ini_localbest(self.parms['list_NumPt4Instance'][0])
            print('Initial Method: Local best.')
        else:
            print('Warrning: Set parms[\'initial_method\'] to be either \'Random\' or \'Local_Best\'.\n')
        self.Zini = Znew
        conti = True
        
        m=len(Znew)
        it_count = 1
        print('Entering %d loops, each operation requires %d comparisons' % (m,self.N))
        while conti:
            Znew,conti,current_fval = self.update2(Znew,self.Xnd)
            
            if conti==1:
                if disp_info:
                    print('Set m=%d, current fval=%6.2f, continue searching.' % (m,current_fval))
            else:
                if disp_info:
                    print('Set m=%d, current fval=%6.2f, which is not getting \
                          better. End of search.' % (m,current_fval))
            it_count +=1
            m = len(Znew)
            if m==5:
                pass
            
            if m==1:
                tmp0 = np.sum(self.dist_xz,axis=0)
                Znew = np.argmin(tmp0)
                conti=0
                
        print('Minimum found when m=%d' % m)
        toc = time.time()-tic
        print('Using %4.2f sec.' % toc)
        uniqueZ = np.unique(Znew)
        
        Z_p,Z_ind = self.Zind2p(uniqueZ)
        cc_ind_wrt_p = [Z_p,Z_ind]
        self.cc_ind_wrt_p=cc_ind_wrt_p
        self.cc_ind = uniqueZ
        self.fvals,self.se = self.target_fun(uniqueZ)
        
        p_ary,id_ary = cc_ind_wrt_p
        X_id_found = [None]*m
        for i in range(m):
            X_id_found[i]=self.list_Allz_Np[p_ary[i]][id_ary[i],:]
        self.list_xid_found = X_id_found
    def Zind_ini_rand(self,SeedNum):
        np.random.seed(SeedNum)
        n = self.n
        dist_xz = self.dist_xz
        Zini = np.zeros((n,),dtype='int')
        for i in range(n):
            di = dist_xz[i,:]
            id_mi = np.nonzero(di==np.min(di))[0]
            if id_mi.size==1:
                zid=id_mi
            else:
                tmp = np.random.randint(0,len(id_mi),1)
                zid=id_mi[tmp]
            Zini[i]=zid
#        Ids = np.arange(n)
#        i = 0
#        while len(Ids)>0:
#            di = dist_xz[Ids[0],:]
#            id_mi = np.nonzero(di==np.min(di))[0]
#            if id_mi.size==1:
#                zid=id_mi
#            else:
#                tmp = np.random.randint(0,len(id_mi),1)
#                zid=id_mi[tmp]
#            Zini[i]=zid
#            p,p_id = self.Zind2p(zid)
#            id_used=self.list_Allz_Np[p][p_id,:]
#            id_used = id_used[0]
#            for j in range(len(id_used)):
#                k = np.nonzero(Ids==id_used[j])[0]
#                Ids = np.delete(Ids,k)
#            i = i+1
#        Zini = np.delete(Zini,np.arange(i,n))
        return Zini

    def Zind_ini_localbest(self,p):
        """
        under construction for usage of multiple models
        """
        Xnd = self.Xnd
        n,d = Xnd.shape        
        NN = np.ceil(n/5)
        distX_nn = dist_square(Xnd)
        Z_Xid = np.zeros((n,p),dtype='int')# each row will be p indexes of Xnd.
        k = nchoosek(NN,p-1)# number of planes to consider for each point
        for i in range(n):
            di = distX_nn[i,:]
            di[i]=np.max(di)+1# just to remove it self.
            sort_di = np.argsort(di)
            id_i= sort_di[0:NN]            
            ids = np.zeros((k,p),dtype='int')
            ids[:,0]=i
            ids[:,1:]=np.vstack(it.combinations(id_i,p-1))
            Zid_i = self.fcn_id_wrt_AllComb(p,ids)
            dist = self.dist_xz[np.ix_(id_i,Zid_i)]# np.ix_:index cross product 
            sdist = np.sum(dist,axis=0)
            if sdist.size!=0:
                id_wrt_Zid_i = np.argmin(sdist)
                Z_Xid[i,:] = ids[id_wrt_Zid_i,:]# each row consisted of p indexes of Xnd.
            else:
                Z_Xid[i,:] = np.random.randint(0,n,p)
        # removing the redundant Zini
        list_Zini=[Z_Xid[0,:]]
        Zini_appear = np.hstack(list_Zini)
        for i in np.arange(1,n):
            Yi = Z_Xid[i,:]
            condi = any([all(Zini_appear!=Yi[j]) for j in range(p)])
            # condi is True if any one of the Yi is not appeared before
            if condi:
                list_Zini.append(Z_Xid[i,:])
                Zini_appear = np.unique(np.hstack(list_Zini))
        Z_Xid_rm = np.vstack(list_Zini)
        Zini = self.fcn_id_wrt_AllComb(p,Z_Xid_rm)        
        return Zini  
        
    def target_fun(self,Zind):
        Zind = np.unique(Zind)
        p_ary,_= self.Zind2p(Zind)         
        prior_term =self.gamma(p_ary)        
        mat_xz = self.sub_mat_xz(self.dist_xz,Zind)
        try:
            minxz  = np.min(mat_xz,axis=1)
        except:# when m=1
            minxz = mat_xz
        se = np.sum(minxz)# se = sum of error
        data_term = se/self.n
        return prior_term+self.lam*data_term,se
        
    def target_fun2(self,Zind):
        
        Zind = np.unique(Zind)
        p_ary,_= self.Zind2p(Zind)         
        prior_term =self.gamma(p_ary)  
        
        mat_xz = self.sub_mat_xz(self.dist_xz,Zind)
        minxz  = np.min(mat_xz,axis=1)
        data_term = np.sum(minxz)/self.n
        fv = prior_term+self.lam*data_term
        return fv,p_ary,data_term,minxz
#    def target_fun_addz(self,pr0,data0,arrminxz,Zinds,Zind_add):        
#        prior_term= pr0+2*len(Zinds)+1
#        xz = self.dist_xz[:,Zind_add]
#
#        tmp2arr = xz-arrminxz
#        tmp2arr = tmp2arr*(tmp2arr<0)
#        dataterm_decrese = np.sum(tmp2arr)/self.n
#        data_term = data0+dataterm_decrese
#        return prior_term+self.lam*data_term
    def data_term_addz(self,arrminxz,Zinds_add):
        """
        arrminxz: (n,) array
        Zinds_add: (K,) int array
        """
        K = len(Zinds_add)
        dataterm_add = np.zeros((K,))
        for k in range(K):
            xz = self.dist_xz[:,Zinds_add[k]]
            tmp2arr = xz-arrminxz
            tmp2arr = tmp2arr[tmp2arr<0]
            dataterm_add[k] = np.sum(tmp2arr)
        return dataterm_add

    def targert_fun_lite(self,p_ary,data0,data1_add_K):
        prior_term=  self.gamma(np.append(p_ary,0))
        data_term = data0+data1_add_K/self.n
        return prior_term+self.lam*data_term
        
    def update2(self,Zind,Xnd):
        """
        Zind is always unique
        """
        All_Zind = np.arange(self.N)
        Zind_del = np.delete(All_Zind,Zind)
        m = len(Zind)
        fval0,_,_,_ = self.target_fun2(Zind)        
        if m==self.N:
            fv_wo_i = np.zeros((m,))
            for el in range(m):
                Zid_wo_i = np.delete(Zind,el)
                fv_wo_i[el],_ = self.target_fun(Zid_wo_i)
            id_min = np.argmin(fv_wo_i)
            if fv_wo_i[id_min]<fval0:
                Zind = np.delete(Zind,id_min)
                current_fval = fv_wo_i[id_min]
                conti = True
        else:
            fv = np.zeros((m,))
#            K = len(Zind_del)
            dmat = self.dist_xz[:,Zind_del]#(n,K)
            Al = np.zeros((m,),dtype = 'int32')
            Idk = np.zeros((m,),dtype = 'int32')
            for el in range(m):
                Zid_wo_i = np.delete(Zind,el)
                fv_wo_i,p_ary0,da0,arr_minxz=self.target_fun2(Zid_wo_i)
#                fv_move_i =np.zeros((K,))
#                data_term_add=self.data_term_addz(arr_minxz,Zind_del)
                
                data_term_add = data_term_addz_C(dmat,arr_minxz)#(K,)
                
                fv_move_i = self.targert_fun_lite(p_ary0,da0,data_term_add)

                id_k = np.argmin(fv_move_i)
                Al[el] = np.argmin([fv_wo_i,fv_move_i[id_k]])
                Idk[el] = id_k
                fv[el] = np.min([fv_wo_i,fv_move_i[id_k]])
            id_min = np.argmin(fv)
            
            if fv[id_min]+1e-4<fval0:
                Zind = np.delete(Zind,id_min)
                current_fval = fv[id_min]
                if Al[id_min]==1:
                    Zind = np.hstack((Zind,Zind_del[Idk[id_min]]))
                conti = True
            else:
                conti = False
                current_fval = fval0
            Zind = np.unique(Zind)
        return Zind,conti,current_fval

        
    def sub_mat_zz(self,A,ind):
        tmp = A[:,ind]
        tmp2 = tmp[ind,:]
        return tmp2
        
    def sub_mat_xz(self, A, ind):
        tmp = A[:,ind]
        return tmp       
    def run_exactsol_with_C(self,m=1):
        """
        m: number of instances
        Only support m=1,2,3
        """
        dist_xz = self.dist_xz #(n,N)
        tic = time.time()
        if m==1:
            res = np.sum(dist_xz,axis=0)
            Zid = np.argmin(res)
            fv_l = res[Zid]
        if m==2:
            Zid = exactsol_m2(dist_xz)
        if m==3:
            Zid = exactsol_m3(dist_xz)
    
        A = dist_xz[:,Zid]
        if m>1:
            res = np.min(A,axis=1)
            fv_l = np.sum(res)
        toc = time.time()-tic
        print('Using %4.2f sec.' % toc)
        
        p_ary,id_ary = self.Zind2p(Zid)
        X_id_found = [None]*m
        for i in range(m):
            X_id_found[i]=self.list_Allz_Np[p_ary[i]][id_ary[i],:]
        return fv_l, X_id_found
    def fcn_classify(self,list_xid_found,list_zeta=None):
    
        Xnd = self.data
        _,data_se = standerdizing(Xnd)
        n,d = Xnd.shape
        
        m = len(list_xid_found)
        distmat = np.zeros((n,m))
        if list_zeta is None:
            list_zeta =[1]*10
        for i in range(m):
            Z_ids = list_xid_found[i]# Z_ids is (p,) array, array of indexes of data
            p = len(Z_ids)
            distmat[:,i] = gen_dist_arr(Xnd,Z_ids,self.beta,list_zeta[p])
        data_cluster_id = np.argmin(distmat,axis=1)
        h = np.sum(np.min(distmat,axis=1))
        h_adj = h/(data_se**self.beta)
        return data_cluster_id,h_adj        
class Pickable(object):
    def __init__(self,a):
        self.parms = a.parms
        self.data =a.data
        self.result = {'list_xid_found':a.list_xid_found,\
                       'fvals': a.fvals,'Zini':a.Zini,'SumOfError':a.se}
def save_result(Class_line_cluster,filename=None):
    b = Pickable(Class_line_cluster)
    dill.dump(b, open(filename+'.rra','wb'))
def load_result(filename):
    c = dill.load(open(filename+'.rra','rb'))
    a = RRalg(c.data,c.parms,c.result)
    return a
    
def Cart2theta(arr_n2):
    """
    arr_n2: size (n,2) array
    
    e.g. x1 = arr_n2[1,:]
         This function find the angle of the following two lines:
             1. (0,0) to x1
             2. x-axis
             
    theta \in [0,2*pi)
    """
    x = arr_n2[:,0]
    y = arr_n2[:,1]
    n = len(x)
    id_x0 = np.nonzero(x==0)[0]
    id_l = np.delete(np.arange(n),id_x0)
    theta = np.zeros((n,))
    theta[id_x0]=np.pi/2.0
    theta[id_l]=np.mod(np.arctan(y[id_l]/x[id_l]),np.pi)
    return theta

def points_proj2line(Xn2,line):
    """
    line: size (3,), format is (x,y,theta), 
        means the line passes (x,y) with angle theta.
    Xn2 = data points with size (n,2)
    """
    X0 = line[:2]
    th = line[-1]
    r0= np.array([np.cos(th),np.sin(th)])
    Xs = Xn2-X0
    r0r0 = np.dot(r0[:,None],r0[None,:])
    proj = np.dot(Xs,r0r0)+X0
    return proj
    
def nchoosek(n,k):
   a = np.math.factorial(n)/(np.math.factorial(n-k)*np.math.factorial(k))
   return int(a)
    
def dist_square(X_nd):
    n,d = X_nd.shape
    dist=np.zeros((n,n))
    for i in range(d):
        z = X_nd[:,i]
        z = z[...,None]
        zz = z*z.T
        zdiag=np.diag(zz)
        z2 = zdiag[...,None]*np.ones([1,n])
        disti = z2+z2.T-2*zz
        dist = dist+disti        
        
    dist[np.nonzero(dist<0)]=0
    return dist

def standerdizing(Y_nd):
    mean = np.mean(Y_nd,axis=0)
    tmp=np.var(Y_nd,axis=0)
    data_se = np.sqrt(np.sum(tmp))
    X_nd=(Y_nd-mean)/data_se
    return X_nd,data_se
def normalize(Xnd):
    c = np.sqrt(np.sum(Xnd**2,axis=1))
    Xnd_normalized = Xnd.T/c
    return Xnd_normalized.T

    
def gen_dist_arr(Xnd,ids,beta,adj_quotient=1):
    d = Xnd.shape[1]
    Xi = Xnd[ids[0],:]
    Yi_dn = (Xnd-Xi).T #(d,n)
    p = len(ids)
    if p==1:
        dist = np.sqrt(np.sum(Yi_dn**2,axis=0))**beta #(n,)
        distarr = dist/adj_quotient
    else:
        A = Yi_dn[:,ids[1:]]#(d,p-1)
        AtAinv = np.linalg.inv(np.dot(A.T,A))#(2,2)
        AtAinv_At = np.dot(AtAinv,A.T)
        M_dd = np.eye(d)-np.dot(A,AtAinv_At)#(d,d)
        proj_dn = np.dot(M_dd,Yi_dn)
        dist = np.sqrt(np.sum(proj_dn**2,axis=0))**beta#(n,)
        distarr = dist/adj_quotient
    return distarr