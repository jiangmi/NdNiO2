import variational_space as vs
import lattice as lat
import bisect
import numpy as np
import scipy.sparse as sps
import parameters as pam
            
def find_singlet_triplet_partner(state,VS):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Applies to general opposite-spin state, not nesessarily in d_double

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    if state['type'] == 'two_hole_no_eh': 
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        
        slabel = [s2,orb1,x1,y1,z1,s1,orb2,x2,y2,z2]
        tmp_state = vs.create_two_hole_no_eh_state(slabel)
        partner_state, phase = vs.make_state_canonical(tmp_state)
        
    return VS.get_index(partner_state), phase

def create_singlet_triplet_basis_change_matrix(VS,d_double):
    '''
    Create a matrix representing the basis change to singlets/triplets. The
    columns of the output matrix are the new basis vectors. 
    The Hamiltonian transforms as U_dagger*H*U. 

    Parameters
    ----------
    phase: dictionary containing the phase factors created with
        hamiltonian.create_phase_dict.
    VS: VariationalSpace class from the module variational_space. Should contain
        only zero-magnon states.

    Returns
    -------
    U: matrix representing the basis change to singlets/triplets in
        sps.coo format.
    '''
    data = []
    row = []
    col = []
    
    #count_upup, count_updn, count_dnup, count_dndn = count_VS(VS)
    #print count_upup, count_updn, count_dnup, count_dndn
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val    = np.zeros(VS.dim, dtype=int)
    Sz_val   = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    for i in range(0,VS.dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        if start_state['type'] == 'two_hole_no_eh': 
            s1 = start_state['hole1_spin']
            s2 = start_state['hole2_spin']
            orb1 = start_state['hole1_orb']
            orb2 = start_state['hole2_orb']

            if i not in count_list:
                j, ph = find_singlet_triplet_partner(start_state,VS)
                #print "partner states:", i,j
                #print "state i = ", s1, orb1, s2, orb2
                #j_state = VS.get_state(VS.lookup_tbl[j])
                #js1 = j_state['hole1_spin']
                #js2 = j_state['hole2_spin']
                #jorb1 = j_state['hole1_orb']
                #jorb2 = j_state['hole2_orb']
                #print "state j = ", js1, jorb1, js2, jorb2
                
                count_list.append(j)

                if j==i:
                    if s1==s2:
                        # must be triplet
                        data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                        S_val[i] = 1
                        if s1=='up':
                            Sz_val[i] = 1
                        elif s1=='dn':
                            Sz_val[i] = -1
                        count_triplet += 1
                    else:
                        # only possible other states for j=i 
                        assert(s1=='up' and s2=='dn' and orb1==orb2)

                        # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                        # instead of e1e1 and e2e2
                        if orb1 not in ['dxz','dyz']:
                            data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                            S_val[i]  = 0
                            Sz_val[i] = 0
                            count_singlet += 1
                        elif orb1==orb2=='dxz':  # no need to consider e2='dyz' case
                            # find e2e2 state:
                            for e2 in d_double:
                                state = VS.get_state(VS.lookup_tbl[e2])
                                orb1 = state['hole1_orb']
                                orb2 = state['hole2_orb']
                                if orb1==orb2=='dyz':
                                    data.append(1.0);  row.append(i);  col.append(i)
                                    data.append(1.0);  row.append(e2); col.append(i)
                                    AorB_sym[i]  = 1
                                    S_val[i]  = 0
                                    Sz_val[i] = 0
                                    count_singlet += 1
                                    data.append(1.0);  row.append(i);  col.append(e2)
                                    data.append(-1.0); row.append(e2); col.append(e2)
                                    AorB_sym[e2] = -1
                                    S_val[e2]  = 0
                                    Sz_val[e2] = 0
                                    count_singlet += 1
                else:
                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);  row.append(i); col.append(i)
                    data.append(-ph);   row.append(j); col.append(i)
                    S_val[i]  = 0
                    Sz_val[i] = 0

                    # append matrix elements for triplet states
                    data.append(1.0);  row.append(i); col.append(j)
                    data.append(ph);  row.append(j); col.append(j)
                    S_val[j]  = 1
                    Sz_val[j] = 0

                    count_singlet += 1
                    count_triplet += 1
                    
    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym
                
def find_singlet_triplet_partner_d_double(state,VS):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Right now only applied for d_double states
    
    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    if state['type'] == 'one_hole_one_eh':
        se = state['e_spin']
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        orbe = state['e_orb']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        xe, ye, ze = state['e_coord']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        
        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,'up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1)
        partner_state,phase = vs.make_state_canonical(tmp_state)
        
    return VS.get_index(partner_state), phase

def create_singlet_triplet_basis_change_matrix_d_double(VS,d_double):
    '''
    Similar to above create_singlet_triplet_basis_change_matrix but only applies
    basis change for d_double states
    
    Note that for three hole state, its partner state must have exactly the same
    spin and positions of L and Nd-electron
    '''
    data = []
    row = []
    col = []
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state in d_double to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val  = np.zeros(VS.dim, dtype=int)
    Sz_val = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    # first set the matrix to be identity matrix (for states not d_double)
    for i in range(0,VS.dim):
        if i not in d_double:
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)
        
    for i in d_double:
        start_state = VS.get_state(VS.lookup_tbl[i])
        itype = start_state['type']
        assert(itype == 'one_hole_one_eh')
        
        se = start_state['e_spin']
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        oe = start_state['e_orb']
        o1 = start_state['hole1_orb']
        o2 = start_state['hole2_orb']
        xe, ye, ze = start_state['e_coord']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']

        epos=[xe, ye, ze]

        # note the following is generic for two types of states
        if s1==s2:
            # must be triplet
            # see case 2 of make_state_canonical in vs.py, namely
            # for same spin states, always order the orbitals
            S_val[i] = 1
            data.append(np.sqrt(2.0));  row.append(i); col.append(i)
            if s1=='up':
                Sz_val[i] = 1
            elif s1=='dn':
                Sz_val[i] = -1
            count_triplet += 1

        elif s1=='dn' and s2=='up':
            print ('Error: d_double cannot have states with s1=dn, s2=up !')
            tstate = VS.get_state(VS.lookup_tbl[i])
            tse = tstate['e_spin']
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            torbe = tstate['e_orb']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            txe, tye, tze = tstate['e_coord']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            print ('Error state', i, tse,torbe,txe,tye,ts1,torb1,tx1,ty1,ts2,torb2,tx2,ty2)
            break

        elif s1=='up' and s2=='dn':
            if o1==o2: 
                if o1!='dxz' and o1!='dyz':
                    data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                    S_val[i]  = 0
                    Sz_val[i] = 0
                    count_singlet += 1
                    
                # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                # instead of e1e1 and e2e2
                elif o1=='dxz':  # no need to consider e2='dyz' case
                    # find e2e2 state:
                    for e2 in d_double:
                        state = VS.get_state(VS.lookup_tbl[e2])
                        jtype = state['type']
                        assert(jtype == 'one_hole_one_eh')
                        
                        jse = state['e_spin']
                        js1 = state['hole1_spin']
                        js2 = state['hole2_spin']
                        joe = state['e_orb']
                        jo1 = state['hole1_orb']
                        jo2 = state['hole2_orb']
                        jxe, jye, jze = state['e_coord']
                        jx1, jy1, jz1 = state['hole1_coord']
                        jx2, jy2, jz2 = state['hole2_coord']

                        jepos=[jxe, jye, jze]

                        if not (jse==se and joe==oe and jepos==epos):
                            continue
                           
                        if jo1==jo2=='dyz':
                            data.append(1.0);  row.append(i);  col.append(i)
                            data.append(1.0);  row.append(e2); col.append(i)
                            AorB_sym[i]  = 1
                            S_val[i]  = 0
                            Sz_val[i] = 0
                            count_singlet += 1
                            data.append(1.0);  row.append(i);  col.append(e2)
                            data.append(-1.0); row.append(e2); col.append(e2)
                            AorB_sym[e2] = -1
                            S_val[e2]  = 0
                            Sz_val[e2] = 0
                            count_singlet += 1

            else:
                if i not in count_list:
                    j, ph = find_singlet_triplet_partner_d_double(start_state,VS)
                    assert(j!=i)

                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);  row.append(i); col.append(i)
                    data.append(ph);  row.append(j); col.append(i)
                    S_val[i]  = 0
                    Sz_val[i] = 0

                    #print "partner states:", i,j
                    #print "state i = ", s1, orb1, s2, orb2
                    #print "state j = ",'up',orb2,'dn',orb1

                    # append matrix elements for triplet states
                    data.append(1.0);  row.append(i); col.append(j)
                    data.append(-ph);   row.append(j); col.append(j)
                    S_val[j]  = 1
                    Sz_val[j] = 0

                    count_list.append(j)

                    count_singlet += 1
                    count_triplet += 1

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym

def print_VS_after_basis_change(VS,d_double,S_val,Sz_val):
    print ('print_VS_after_basis_change for d_double states:')
    for i in d_double:
        state = VS.get_state(VS.lookup_tbl[i])
        assert(state['type'] == 'one_hole_one_eh')
        
        se = state['e_spin']
        orbe = state['e_orb']
        xe, ye, ze = state['e_coord']
        ts1 = state['hole1_spin']
        ts2 = state['hole2_spin']
        torb1 = state['hole1_orb']
        torb2 = state['hole2_orb']
        tx1, ty1, tz1 = state['hole1_coord']
        tx2, ty2, tz2 = state['hole2_coord']
        
        print (i, se,orbe,xe,ye,ze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,'S=',S_val[i],'Sz=',Sz_val[i])
            
