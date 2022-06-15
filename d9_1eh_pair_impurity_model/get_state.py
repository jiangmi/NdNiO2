'''
Get state index for A(w)
'''
import subprocess
import os
import sys
import time
import shutil
import numpy

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util

#####################################################################
def get_d9_state_indices(VS):
    Norb = pam.Norb
    dim = VS.dim
    d9_state_indices = []; d9_state_labels = []; d9_labels = []

    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'one_hole_one_eh':
            continue
            
        s1 = state['hole_spin']
        o1 = state['hole_orb']
        
        if s1=='up':
            if o1=='d3z2r2':
                d9_state_indices.append(i); d9_state_labels.append('$d_{z^2}$'); d9_labels.append('dz2')
                print ("d9_state_indices", i, "state: orb= ", o1, "spin= ", s1)
            elif o1=='dx2y2':
                d9_state_indices.append(i); d9_state_labels.append('$d_{x^2-y^2}$'); d9_labels.append('dx2y2')
                print ("d9_state_indices", i, "state: orb= ", o1, "spin= ", s1)
            elif o1=='dxy':
                d9_state_indices.append(i); d9_state_labels.append('$d_{xy}$'); d9_labels.append('dxy')
                print ("d9_state_indices", i, "state: orb= ", o1, "spin= ", s1)
            elif o1=='dxz':
                d9_state_indices.append(i); d9_state_labels.append('$d_{xz/yz}$'); d9_labels.append('dxzyz')
                print ("d9_state_indices", i, "state: orb= ", o1, "spin= ", s1)
                 
    return d9_state_indices, d9_state_labels, d9_labels

def get_d10L_state_indices(VS):
    Norb = pam.Norb
    dim = VS.dim
    d10L_state_indices = []; d10L_state_labels = []

    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype == 'one_hole_one_eh':
            continue
            
        s1 = state['hole_spin']
        o1 = state['hole_orb']
        x1, y1, z1 = state['hole_coord']
        
        if not (abs(x1-1.0)<1.e-3 and abs(y1)<1.e-3):
            continue
        
        if s1=='up':
            if o1 in pam.O_orbs:
                d10L_state_indices.append(i); d10L_state_labels.append(str(o1))
                print ("d10L_state_indices", i, "state: orb= ", o1, "spin= ", s1)
                 
    return d10L_state_indices, d10L_state_labels

#####################################################################
def get_d8s_state_indices(VS, d_double, S_val, Sz_val, AorB_sym, A):
    '''
    Get some special d8s
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8s_state_indices = []; d8s_state_labels = []

    for i in d_double:
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        assert(itype == 'one_hole_one_eh')
            
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        xe, ye, ze = state['e_coord']
        
        if not (xe==0 and ye==0 and ze==1):
            continue
       
        o12 = sorted([o1,o2])
        o12 = tuple(o12)

        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
            
        if o12==('d3z2r2','d3z2r2'):
            if S12==0:
                d8s_state_indices.append(i); d8s_state_labels.append('$S=0, d^8_{z^2z^2}s$'+', $e=$'+str(se))
                print ("d8s_state_indices", i, 'se=', se, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)

        if o12==('d3z2r2','dx2y2'):
            if S12==0:
                #sss = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,'up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1)
                #idx = VS.get_index(sss)
                
                d8s_state_indices.append(i); d8s_state_labels.append('$S=0, d^8_{z^2,x^2-y^2}s$'+', $S_z=$'+str(Sz12)+', $e=$'+str(se))
                print ("d8s_state_indices", i, 'se=', se, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2, "spin= ", s1,s2)
            if S12==1:# and Sz12==0:
                d8s_state_indices.append(i); d8s_state_labels.append('$S=1, d^8_{z^2,x^2-y^2}s$'+', $S_z=$'+str(Sz12)+', $e=$'+str(se))
                print ("d8s_state_indices", i, 'se=', se, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2, "spin= ", s1,s2)
                
        # Special syms without b1 orbital:
        #if (sym=='1B2' or sym=='3B2') and 'd3z2r2' in o12 and 'dxy' in o12:
        #    dd_state_indices.append(i)
        #    print ("d8_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)
              
    return d8s_state_indices, d8s_state_labels

def get_d8s_state_indices_sym(VS, sym, d_double, S_val, Sz_val, AorB_sym, A):
    '''
    Get d8 state index including one dx2y2 hole
    Note: for transformed basis of singlet/triplet
    index can differ from that in original basis
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8s_state_indices = []
    
    # get info specific for particular sym
    state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(A, sym)
    sym_orbs = state_order.keys()

    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        assert(itype == 'one_hole_one_eh')
            
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        xe, ye, ze = state['e_coord']
       
        if not (xe==0 and ye==0 and ze==1):
            continue
            
        o12 = sorted([o1,o2])
        o12 = tuple(o12)

        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
        
        # continue only if (o1,o2) is within desired sym
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        # distinguish (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
        if (o1==o2=='dxz' or o1==o2=='dyz') and AorB_sym[i]!=AorB:
            continue
            
        if 'dx2y2' in o12:
            d8s_state_indices.append(i)
            print ("d8s_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)

        # Special syms without b1 orbital:
        #if (sym=='1B2' or sym=='3B2') and 'd3z2r2' in o12 and 'dxy' in o12:
        #    dd_state_indices.append(i)
        #    print ("d8_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)
              
    return d8s_state_indices

def get_d9Ls_state_indices(VS, S_val, Sz_val):
    '''
    Get d9L state index including one dx2y2 or dz2 hole
    '''    
    Norb = pam.Norb
    dim = VS.dim
    b1Ls_state_indices = []; b1Ls_state_labels = []
    a1Ls_state_indices = []; a1Ls_state_labels = []

    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        
        if itype!='one_hole_one_eh':
            continue
            
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        xe, ye, ze = state['e_coord']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']

        if not (xe==0 and ye==0 and ze==1):
            continue
            
        nNi, nO, dorbs, porbs = util.get_statistic_2orb(o1,o2)
        
        if not (nNi==1 and nO==1):
            continue
        
        # L not far away from Ni impurity
        orbs = [o1,o2]
        xs = [x1,x2]
        ys = [y1,y2]
        idx = orbs.index(porbs[0])
            
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
        
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
        
        # continue only singlet
        if S12!=0:
            continue
            
        if dorbs[0]=='dx2y2':
            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                b1Ls_state_indices.append(i); b1Ls_state_labels.append('$d^9_{x^2-y^2}Ls$')
                print ("b1Ls_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)

        if dorbs[0]=='d3z2r2':
            # for triplet, only need one Sz state; other Sz states have the same A(w)
            if Sz12==0:
                a1Ls_state_indices.append(i); a1Ls_state_labels.append('$d^9_{z^2}Ls$')
                print ("a1Ls_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2)
                 
    return b1Ls_state_indices, b1Ls_state_labels, a1Ls_state_indices, a1Ls_state_labels
