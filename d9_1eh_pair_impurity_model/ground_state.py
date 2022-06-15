import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util

def get_ground_state(matrix, VS, S_val,Sz_val):  
    '''
    Obtain the ground state info
    '''        
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#    M_dense = matrix.todense()
#    print M_dense
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                
#    vals, vecs = np.linalg.eigh(M_dense)
#    vals.sort()
#    print 'lowest eigenvalue of H from np.linalg.eigh = '
#    print vals
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    # get state components in GS; note that indices is a tuple
    for k in range(0,1):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.01)
        
        wgt_b1=0.0; wgt_a1=0.0; wgt_d9_other=0.0; wgt_d10L=0.0; wgt_d10Lp=0.0; 
        wgt_a1b1S1_s=0.0; wgt_a1b1S0_s=0.0; wgt_a1a1S1_s=0.0; wgt_a1a1S0_s=0.0; wgt_b1b1S1_s=0.0; 
        wgt_b1b1S0_s=0.0; wgt_d8s_other=0.0; 
        wgt_a1Ls=0.0; wgt_b1Ls=0.0; wgt_d9Ls_other=0.0; wgt_Lp=0.0; wgt_ep=0.0

        print ("Compute the weights in GS (lowest Aw peak)")

        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            weight = abs(vecs[i,k])**2
            
            if state['type'] == 'one_hole_no_eh':
                s1 = state['hole_spin']
                o1 = state['hole_orb']
                x1,y1,z1 = state['hole_coord']
            
                if i in indices[0]:
                    print ('no e-h state ', s1,o1,x1,y1,z1, ", weight = ", weight)
                 
                if o1=='dx2y2':
                    wgt_b1 += weight
                elif o1=='d3z2r2':
                    wgt_a1 += weight
                elif o1 in pam.Ni_orbs:
                    wgt_d9_other += weight
                elif o1 in pam.O_orbs:
                    if abs(x1)>1. or abs(y1)>1.:
                        wgt_d10Lp += weight
                    else:
                        wgt_d10L  += weight
                        
            if state['type'] == 'one_hole_one_eh':
                se = state['e_spin']
                orbe = state['e_orb']
                xe, ye, ze = state['e_coord']
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']

                S12  = S_val[i]
                Sz12 = Sz_val[i]
                
                o12 = sorted([orb1,orb2])
                o12 = tuple(o12)
                
                nNi, nO, dorbs, porbs = util.get_statistic_2orb(orb1,orb2)
                
                if i in indices[0]:
                    if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.):
                        print ('one e-h state ', se,orbe,xe,ye,ze,s1,orb1,x1,y1,s2,orb2,x2,y2, \
                               'S=',S12, 'Sz=',Sz12, ", weight = ", weight)

                if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(xe)>0. or abs(ye)>0.):
                    # d8s
                    if o12==('d3z2r2','dx2y2'):
                        if S12==1:
                            wgt_a1b1S1_s += weight
                        elif S12==0:
                            wgt_a1b1S0_s += weight
                    elif o12==('dx2y2','dx2y2'):
                        if S12==1:
                            wgt_b1b1S1_s += weight
                        elif S12==0:
                            wgt_b1b1S0_s += weight
                    elif o12==('d3z2r2','d3z2r2'):
                        if S12==1:
                            wgt_a1a1S1_s += weight
                        elif S12==0:
                            wgt_a1a1S0_s += weight
                    elif nNi==2:
                        wgt_d8s_other += weight
                    
                    elif nNi==1:
                        if dorbs[0]=='d3z2r2':
                            wgt_a1Ls += weight
                        elif dorbs[0]=='dx2y2':
                            wgt_b1Ls += weight
                        else:
                            wgt_d9Ls_other += weight

                elif (xe,ye)==(0,0) and (abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.):
                    wgt_Lp += weight
                        
                elif (xe,ye)!=(0,0) and \
                    not (abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.):
                    wgt_ep += weight
                
        print('wgt_b1 = ',wgt_b1)
        print('wgt_a1 = ',wgt_a1)
        print('wgt_d9_other = ',wgt_d9_other)
        print('wgt_d10L = ',wgt_d10L)        
        print('wgt_d10Lp = ',wgt_d10Lp)

        print('wgt_a1b1S1_s = ',wgt_a1b1S1_s)
        print('wgt_a1b1S0_s = ',wgt_a1b1S0_s)
        print('wgt_a1a1S1_s = ',wgt_a1a1S1_s)
        print('wgt_a1a1S0_s = ',wgt_a1a1S0_s)
        print('wgt_b1b1S1_s = ',wgt_b1b1S1_s)
        print('wgt_b1b1S0_s = ',wgt_b1b1S0_s)
        print('wgt_d8s_other = ',wgt_d8s_other)
        
        print('wgt_a1Ls = ',wgt_a1Ls)
        print('wgt_b1Ls = ',wgt_b1Ls)
        print('wgt_d9Ls_other = ',wgt_d9Ls_other)

        print('wgt_Lp = ',wgt_Lp)
        print('wgt_ep = ',wgt_ep)

        print(wgt_b1, wgt_a1, wgt_d9_other, wgt_d10L, wgt_d10Lp, \
              wgt_a1b1S1_s, wgt_a1b1S0_s, wgt_a1a1S1_s, wgt_a1a1S0_s, wgt_b1b1S1_s, wgt_b1b1S0_s, wgt_d8s_other, \
              wgt_a1Ls, wgt_b1Ls, wgt_d9Ls_other, wgt_Lp, wgt_ep)

        print('total weight = ', wgt_b1+ wgt_a1+ wgt_d9_other+ wgt_d10L+ wgt_d10Lp+ \
              wgt_a1b1S1_s+ wgt_a1b1S0_s+ wgt_a1a1S1_s+ wgt_a1a1S0_s+ wgt_b1b1S1_s+ wgt_b1b1S0_s +\
              wgt_d8s_other+ wgt_a1Ls+ wgt_b1Ls+wgt_d9Ls_other+ wgt_Lp+ wgt_ep)

        
    return vals

