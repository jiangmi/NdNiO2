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
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
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
    
    # get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.01)
        wgt_d8 = np.zeros(6)
        wgt_d9L = np.zeros(4)
        wgt_d10L2 = np.zeros(1)

        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            
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
                    print ('no e-h state ', o12[0],x1,y1,z1,o12[1],x2,y2,z2, 'S=',S12,'Sz=',Sz12, \
                      ", weight = ", abs(vecs[i,k])**2)
                        
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

                #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
                #    continue
                if i in indices[0]:
                    print ('one e-h state ', se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2,s3,orb3,x3,y3, \
                          ", weight = ", abs(vecs[i,k])**2)

                # record the weights of 1A1 and 3B1 states a1a1, b1b1, ..., a1b1 in G.S.
#                 if o12[0]==o12[1]=='d3z2r2':
#                     wgt_d8[0] += abs(vecs[i,k])**2
#                 if o12==('dx2y2','dx2y2'):
#                     wgt_d8[1] += abs(vecs[i,k])**2
#                 if o12[0]==o12[1]=='dxy':
#                     wgt_d8[2] += abs(vecs[i,k])**2
#                 if o12[0]=='d3z2r2' and o12[1]=='dx2y2':
#                     wgt_d8[4] += abs(vecs[i,k])**2
#                 if o12[0]=='d3z2r2' and o12[1]=='dxy':
#                     wgt_d8[5] += abs(vecs[i,k])**2

#                 if o12[0]=='d3z2r2' and o12[1] in pam.O_orbs:
#                     wgt_d9L[0] += abs(vecs[i,k])**2
#                 if o12[0]=='dx2y2'  and o12[1] in pam.O_orbs:
#                     wgt_d9L[1] += abs(vecs[i,k])**2
#                 if o12[0]=='dxy'    and o12[1] in pam.O_orbs:
#                     wgt_d9L[2] += abs(vecs[i,k])**2
#                 if (o12[0]=='dxz' or o12[0]=='dyz') and o12[1] in pam.O_orbs:
#                     wgt_d9L[3] += abs(vecs[i,k])**2

#                 if o12[0] in pam.O_orbs and o12[1] in pam.O_orbs:
#                     wgt_d10L2[0] += abs(vecs[i,k])**2

    return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
