import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import time

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util

def get_ground_state(matrix, VS, S_val,Sz_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     #print 'H='
#     #print M_dense
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()
#     print 'lowest eigenvalue of H from np.linalg.eigh = '
#     print vals
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    print("---get_ground_state_eigsh %s seconds ---" % (time.time() - t1))
    
    # get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.01)
        wgt_d8_b1b1 = 0.0; wgt_d8_other = 0.0
        wgt_b1L = 0.0; wgt_b1Lp = 0.0; wgt_d9L_other = 0.0
        wgt_d10L2 = 0.0; wgt_d10L2p = 0.0
        wgt_a1b1S1_Ls = 0.0; wgt_a1b1S0_Ls = 0.0
        wgt_a1a1S1_Ls = 0.0; wgt_a1a1S0_Ls = 0.0
        wgt_b1b1S1_Ls = 0.0; wgt_b1b1S0_Ls = 0.0
        wgt_d8Ls_other = 0.0
        wgt_a1L2s = 0.0; wgt_b1L2s = 0.0; wgt_d9L2s_other = 0.0
        wgt_Lp = 0.0; wgt_ep = 0.0

        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            
            weight = abs(vecs[i,k])**2
            
            if state['type'] == 'two_hole_no_eh':
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                
                # also obtain the total S and Sz of the state
                S12  = S_val[i]
                Sz12 = Sz_val[i]

                o12 = sorted([orb1,orb2])
                o12 = tuple(o12)
            
                if i in indices[0]:
                    if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.):
                        print ('no e-h state ', i, s1,orb1,x1,y1,s2,orb2,x2,y2, 'S=',S12,'Sz=',Sz12, \
                                ", weight = ", weight)
                        
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
                    if o12==('dx2y2','dx2y2'):
                        wgt_d8_b1b1 += weight
                    else:
                        wgt_d8_other += weight

                elif o12[0] in pam.Ni_orbs and o12[1] in pam.O_orbs:
                    if o12[0]=='dx2y2':
                        if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
                            wgt_b1Lp += weight
                        else:
                            wgt_b1L  += weight
                    else:
                        wgt_d9L_other += weight

                elif o12[0] in pam.O_orbs and o12[1] in pam.O_orbs:
                    if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
                        wgt_d10L2p += weight
                    else:
                        wgt_d10L2  += weight

            if state['type'] == 'two_hole_one_eh':
                se = state['e_spin']
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                s3 = state['hole3_spin']
                orbe = state['e_orb']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                orb3 = state['hole3_orb']
                xe, ye, ze = state['e_coord']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                x3, y3, z3 = state['hole3_coord']
                
                S12  = S_val[i]
                Sz12 = Sz_val[i]
                
                o12 = sorted([orb1,orb2])
                o12 = tuple(o12)
                o23 = sorted([orb2,orb3])
                o23 = tuple(o23)
                o13 = sorted([orb1,orb3])
                o13 = tuple(o13)
                
                nNi, nO, dorbs, porbs = util.get_statistic_3orb(orb1,orb2,orb3)
                
                #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
                #    continue
                
                if i in indices[0]:
                    if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.):
                        print ('one e-h state ', i, se,orbe,xe,ye,ze,s1,orb1,x1,y1,s2,orb2,x2,y2,s3,orb3,x3,y3, \
                               'S=',S12,'Sz=',Sz12, ", weight = ", weight)

                if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.\
                       or abs(xe)>0. or abs(ye)>0.):
                    # d8Ls
                    if nNi==2:
                        if o12==('d3z2r2','dx2y2') or o23==('d3z2r2','dx2y2') or o13==('d3z2r2','dx2y2'):
                            if S12==1:
                                wgt_a1b1S1_Ls += weight
                            elif S12==0:
                                wgt_a1b1S0_Ls += weight
                        elif o12==('dx2y2','dx2y2') or o23==('dx2y2','dx2y2') or o13==('dx2y2','dx2y2'):
                            if S12==1:
                                wgt_b1b1S1_Ls += weight
                            elif S12==0:
                                wgt_b1b1S0_Ls += weight
                        elif o12==('d3z2r2','d3z2r2') or o23==('d3z2r2','d3z2r2') or o13==('d3z2r2','d3z2r2'):
                            if S12==1:
                                wgt_a1a1S1_Ls += weight
                            elif S12==0:
                                wgt_a1a1S0_Ls += weight
                        else:
                            wgt_d8Ls_other += weight

                    elif nNi==1:
                        if dorbs[0]=='d3z2r2':
                            wgt_a1L2s += weight
                        elif dorbs[0]=='dx2y2':
                            wgt_b1L2s += weight
                        else:
                            wgt_d9L2s_other += weight

                elif (xe,ye)==(0,0) and (abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.):
                    wgt_Lp += weight
                        
                elif (xe,ye)!=(0,0) and \
                    not (abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.):
                    wgt_ep += weight
                
    print('wgt_d8_b1b1 = ',wgt_d8_b1b1)
    print('wgt_d8_other = ',wgt_d8_other)
    print('wgt_b1L = ',wgt_b1L)
    print('wgt_b1Lp = ',wgt_b1Lp)
    print('wgt_d9L_other = ',wgt_d9L_other)
    print('wgt_d10L2 = ',wgt_d10L2)
    print('wgt_d10L2p = ',wgt_d10L2p)
    
    print('wgt_a1b1S1_Ls = ',wgt_a1b1S1_Ls)
    print('wgt_a1b1S0_Ls = ',wgt_a1b1S0_Ls)
    print('wgt_a1a1S1_Ls = ',wgt_a1a1S1_Ls)
    print('wgt_a1a1S0_Ls = ',wgt_a1a1S0_Ls)
    print('wgt_b1b1S1_Ls = ',wgt_b1b1S1_Ls)
    print('wgt_b1b1S0_Ls = ',wgt_b1b1S0_Ls)
    print('wgt_d8Ls_other = ',wgt_d8Ls_other)
    print('wgt_a1L2s = ',wgt_a1L2s)
    print('wgt_b1L2s = ',wgt_b1L2s)
    print('wgt_d9L2s_other = ',wgt_d9L2s_other)
    
    print('wgt_Lp = ',wgt_Lp)
    print('wgt_ep = ',wgt_ep)
    
    print(wgt_d8_b1b1, wgt_d8_other, wgt_b1L, wgt_b1Lp, wgt_d9L_other, wgt_d10L2, wgt_d10L2p, \
          wgt_a1b1S1_Ls, wgt_a1b1S0_Ls, wgt_a1a1S1_Ls, wgt_a1a1S0_Ls, wgt_b1b1S1_Ls, wgt_b1b1S0_Ls, wgt_d8Ls_other, \
          wgt_a1L2s, wgt_b1L2s, wgt_d9L2s_other, wgt_Lp, wgt_ep)
    
    print('total weight = ', wgt_d8_b1b1+ wgt_d8_other+wgt_b1L+ wgt_b1Lp+ wgt_d9L_other+ wgt_d10L2+ wgt_d10L2p+ \
          wgt_a1b1S1_Ls+ wgt_a1b1S0_Ls+ wgt_a1a1S1_Ls+ wgt_a1a1S0_Ls+ wgt_b1b1S1_Ls+ wgt_b1b1S0_Ls +\
          wgt_d8Ls_other+ wgt_a1L2s+ wgt_b1L2s+wgt_d9L2s_other+ wgt_Lp+ wgt_ep)
    
    return vals #, vecs, wgt_d8, wgt_d9L, wgt_d10L2
