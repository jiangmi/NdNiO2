'''
Functions for constructing individual parts of the Hamiltonian. The 
matrices still need to be multiplied with the appropriate coupling 
constants t_pd, t_pp, etc..

If adding into diagonal Nd atoms (with only s-like orbital as approximation)
then only dxy hops with it
'''
import parameters as pam
import lattice as lat
import variational_space as vs 
import utility as util
import numpy as np
import scipy.sparse as sps

directions_to_vecs = {'UR': (1,1,0),\
                      'UL': (-1,1,0),\
                      'DL': (-1,-1,0),\
                      'DR': (1,-1,0),\
                      'L': (-1,0,0),\
                      'R': (1,0,0),\
                      'U': (0,1,0),\
                      'D': (0,-1,0),\
                      'T': (0,0,1),\
                      'B': (0,0,-1),\
                      'L2': (-2,0,0),\
                      'R2': (2,0,0),\
                      'U2': (0,2,0),\
                      'D2': (0,-2,0),\
                      'T2': (0,0,2),\
                      'B2': (0,0,-2),\
                      'pzL': (-1,0,1),\
                      'pzR': (1,0,1),\
                      'pzU': (0,1,1),\
                      'pzD': (0,-1,1),\
                      'mzL': (-1,0,-1),\
                      'mzR': (1,0,-1),\
                      'mzU': (0,1,-1),\
                      'mzD': (0,-1,-1)}
tpp_nn_hop_dir = ['UR','UL','DL','DR']
tOs_nn_hop_dir = ['U2','D2','R2','L2','T2','B2']

def set_tpd_tpp_tNiOs(Norb,tpd,tpp,pds,pdp,pps,ppp,tNiOs):
    # dxz and dyz has no tpd hopping
    if pam.Norb==8:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==10 or pam.Norb==11:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D']}
    elif pam.Norb==12:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'dxz'   : ['L','R'],\
                          'dyz'   : ['U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'pz1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D'],\
                          'pz2'   : ['U','D']}
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==11 or pam.Norb==12:
        tNiOs_nn_hop_dir = {'d3z2r2': ['T','B'],\
                            'Os'    : ['T','B']}
        
    if pam.Norb==8:
        tpd_orbs = {'d3z2r2','dx2y2','px','py'}
    elif pam.Norb==10:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','px1','py1','px2','py2'}
    elif pam.Norb==12:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','dxz','dyz','px1','py1','pz1','px2','py2','pz2'}
        
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==12:
        tNiOs_orbs  = {'d3z2r2','Os'}
        
    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    # Or see George's email on Aug.19, 2021:
    # dx2-y2 hoping to the O px in the positive x direction should be positive for electrons and for O in the minus x
    # directions should be negative for electrons, i.e. the hoping integral should be minus sign of overlap integral
    # between two neighboring atoms. 
    if pam.Norb==8:
        # d3z2r2 has +,-,+ sign structure so that it is - in x-y plane
        tpd_nn_hop_fac = {('d3z2r2','L','px'): -tpd/np.sqrt(3),\
                          ('d3z2r2','R','px'):  tpd/np.sqrt(3),\
                          ('d3z2r2','U','py'):  tpd/np.sqrt(3),\
                          ('d3z2r2','D','py'): -tpd/np.sqrt(3),\
                          ('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','L','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','D','d3z2r2'):  tpd/np.sqrt(3),\
                          ('py','U','d3z2r2'): -tpd/np.sqrt(3),\
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
    elif pam.Norb==10:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -pds/2.0,\
                          ('d3z2r2','R','px1'):  pds/2.0,\
                          ('d3z2r2','U','py2'):  pds/2.0,\
                          ('d3z2r2','D','py2'): -pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -pds/2.0,\
                          ('px1','L','d3z2r2'):  pds/2.0,\
                          ('py2','D','d3z2r2'):  pds/2.0,\
                          ('py2','U','d3z2r2'): -pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp}
    elif pam.Norb==12:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -pds/2.0,\
                          ('d3z2r2','R','px1'):  pds/2.0,\
                          ('d3z2r2','U','py2'):  pds/2.0,\
                          ('d3z2r2','D','py2'): -pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          ('dxz','L','pz1'):  -pdp,\
                          ('dxz','R','pz1'):   pdp,\
                          ('dyz','U','pz2'):   pdp,\
                          ('dyz','D','pz2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -pds/2.0,\
                          ('px1','L','d3z2r2'):  pds/2.0,\
                          ('py2','D','d3z2r2'):  pds/2.0,\
                          ('py2','U','d3z2r2'): -pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp,\
                          ('pz1','R','dxz'):  -pdp,\
                          ('pz1','L','dxz'):   pdp,\
                          ('pz2','D','dyz'):   pdp,\
                          ('pz2','U','dyz'):  -pdp}
    ########################## tpp below ##############################
    if pam.Norb==8:
        tpp_nn_hop_fac = {('UR','px','py'): -tpp,\
                          ('UL','px','py'):  tpp,\
                          ('DL','px','py'): -tpp,\
                          ('DR','px','py'):  tpp}
    elif pam.Norb==10:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps)}
    elif pam.Norb==12:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps),\
                          ('UR','pz1','pz2'):  ppp,\
                          ('UL','pz1','pz2'):  ppp,\
                          ('DL','pz1','pz2'):  ppp,\
                          ('DR','pz1','pz2'):  ppp}
        
    ########################## tNiOs below ##############################
    # O vacancy which only hops to Ni's d3z2-r2
    # See George's emails on Dec.5, 2020
    if pam.Norb==8 or pam.Norb==10 or pam.Norb==12:
        tNiOs_nn_hop_fac = {('Os','T', 'd3z2r2'): tNiOs,\
                            ('B','Os', 'd3z2r2'): tNiOs}
        
    return tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, \
           tNiOs_nn_hop_dir, tNiOs_orbs, tNiOs_nn_hop_fac, tpp_nn_hop_fac
        
def get_interaction_mat(A, sym):
    '''
    Get d-d Coulomb and exchange interaction matrix
    total_spin based on lat.spin_int: up:1 and dn:0
    
    Rotating by 90 degrees, x goes to y and indeed y goes to -x so that this basically interchanges 
    spatial wave functions of two holes and can introduce - sign (for example (dxz, dyz)).
    But one has to look at what such a rotation does to the Slater determinant state of two holes.
    Remember the triplet state is (|up,down> +|down,up>)/sqrt2 so in interchanging the holes 
    the spin part remains positive so the spatial part must be negative. 
    For the singlet state interchanging the electrons the spin part changes sign so the spatial part can stay unchanged.
    
    Triplets cannot occur for two holes in the same spatial wave function while only singlets only can
    But both singlets and triplets can occur if the two holes have orthogonal spatial wave functions 
    and they will differ in energy by the exchange terms
    
    ee denotes xz,xz or xz,yz depends on which integral <ab|1/r_12|cd> is nonzero, see handwritten notes
    
    AorB_sym = +-1 is used to label if the state is (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
    For syms (in fact, all syms except 1A1 and 1B1) without the above two states, AorB_sym is set to be 0
    
    Here different from all other codes, change A by A/2 to decrease the d8 energy to some extent
    the remaining A/2 is shifted to d10 states, see George's email on Jun.21, 2021
    and also set_edepeOs subroutine
    '''
    B = pam.B
    C = pam.C
    
    # not useful if treat 1A1 and 1B1 as correct ee states as (exex +- eyey)/sqrt(2)
    if sym=='1AB1':
        fac = np.sqrt(6)
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 4,\
                       ('d3z2r2','dx2y2') : 5}
        interaction_mat = [[A/2.+4.*B+3.*C,  4.*B+C,       4.*B+C,           B+C,           B+C,       0], \
                           [4.*B+C,       A/2.+4.*B+3.*C,  C,             3.*B+C,        3.*B+C,       0], \
                           [4.*B+C,       C,            A/2.+4.*B+3.*C,   3.*B+C,        3.*B+C,       0], \
                           [B+C,          3.*B+C,       3.*B+C,        A/2.+4.*B+3.*C,   3.*B+C,       B*fac], \
                           [B+C,          3.*B+C,       3.*B+C,        3.*B+C,        A/2.+4.*B+3.*C, -B*fac], \
                           [0,            0,            0,              B*fac,         -B*fac,      A/2.+2.*C]]
    if sym=='1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 3}
        interaction_mat = [[A/2.+4.*B+3.*C,  4.*B+C,       4.*B+C,        fac*(B+C)], \
                           [4.*B+C,       A/2.+4.*B+3.*C,  C,             fac*(3.*B+C)], \
                           [4.*B+C,       C,            A/2.+4.*B+3.*C,   fac*(3.*B+C)], \
                           [fac*(B+C),    fac*(3.*B+C), fac*(3.*B+C),  A/2.+7.*B+4.*C]]
    if sym=='1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dx2y2'): 0,\
                       ('dxz','dxz')     : 1,\
                       ('dyz','dyz')     : 1}
        interaction_mat = [[A/2.+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A/2.+B+2.*C]]
    if sym=='1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0}
        interaction_mat = [[A/2.+4.*B+2.*C]]
    if sym=='3A2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0,\
                       ('dxz','dyz')  : 1}
        interaction_mat = [[A/2.+4.*B,   6.*B], \
                           [6.*B,     A/2.-5.*B]]
    if sym=='3B1':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('d3z2r2','dx2y2'): 0}
        interaction_mat = [[A/2.-8.*B]]
    if sym=='1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxy'): 0,\
                       ('dxz','dyz')   : 1}
        interaction_mat = [[A/2.+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A/2.+B+2.*C]]
    if sym=='3B2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = -1
        state_order = {('d3z2r2','dxy'): 0}
        interaction_mat = [[A/2.-8.*B]]
    if sym=='1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}    
        interaction_mat = [[A/2.+3.*B+2.*C,  0,           -B*fac,      0,          0,        -B*fac], \
                           [0,            A/2.+3.*B+2.*C,  0,          B*fac,     -B*fac,     0], \
                           [-B*fac,       0,            A/2.+B+2.*C,   0,          0,        -3.*B], \
                           [0,            B*fac,        0,          A/2.+B+2.*C,   3.*B,      0 ], \
                           [0,           -B*fac,        0,          3.*B,       A/2.+B+2.*C,  0], \
                           [-B*fac,       0,           -3.*B,       0,          0,         A/2.+B+2.*C]]
    if sym=='3E':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}        
        interaction_mat = [[A/2.+B,         0,         -3.*B*fac,    0,          0,        -3.*B*fac], \
                           [0,           A/2.+B,        0,           3.*B*fac,  -3.*B*fac,  0], \
                           [-3.*B*fac,   0,          A/2.-5.*B,      0,          0,         3.*B], \
                           [0,           3.*B*fac,   0,           A/2.-5.*B,    -3.*B,      0 ], \
                           [0,          -3.*B*fac,   0,          -3.*B,       A/2.-5.*B,    0], \
                           [-3.*B*fac,   0,          3.*B,        0,          0,         A/2.-5.*B]]
        
    return state_order, interaction_mat, Stot, Sz_set, AorB_sym

def set_matrix_element(row,col,data,new_state,col_index,VS,element):
    '''
    Helper function that is used to set elements of a matrix using the
    sps coo format.

    Parameters
    ----------
    row: python list containing row indices
    col: python list containing column indices
    data: python list containing non-zero matrix elements
    col_index: column index that is to be appended to col
    new_state: new state corresponding to the row index that is to be
        appended.
    VS: VariationalSpace class from the module variationalSpace
    element: (complex) matrix element that is to be appended to data.

    Returns
    -------
    None, but appends values to row, col, data.
    '''
    row_index = VS.get_index(new_state)
    if row_index != None:
        data.append(element)
        row.append(row_index)
        col.append(col_index)

def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) pd hopping part of the Hamiltonian
    Only hole can hop with tpd

    Parameters
    ----------
    VS: VariationalSpace class from the module variationalSpace
    
    Returns
    -------
    matrix: (sps coo format) t_pd hopping part of the Hamiltonian without 
        the prefactor t_pd.
    
    Note from the sps documentation
    -------------------------------
    By default when converting to CSR or CSC format, duplicate (i,j)
    entries will be summed together
    '''    
    print ("start create_tpd_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        
        # only hole hops with tpd
        if start_state['type'] == 'one_hole_no_eh':
            s1 = start_state['hole_spin']
            orb1 = start_state['hole_orb']
            x1,y1,z1 = start_state['hole_coord']

            # hole hops: some d-orbitals might have no tpd
            if orb1 in tpd_orbs:
                for dir_ in tpd_nn_hop_dir[orb1]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1 == ['NotOnSublattice']:
                        continue
                        
                    # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                    for o1 in orbs1:
                        if o1 not in tpd_orbs:
                            continue

                        tmp_state = vs.create_one_hole_no_eh_state(s1,o1,x1+vx,y1+vy,z1+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)
                        #new_state,ph = vs.make_state_canonical_old(tmp_state)

                        o12 = tuple([orb1, dir_, o1])
                        if o12 in tpd_keys:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)
          
        if start_state['type'] == 'one_hole_one_eh':
            se = start_state['e_spin']
            s1 = start_state['hole1_spin']
            s2 = start_state['hole2_spin']
            orbe = start_state['e_orb']
            orb1 = start_state['hole1_orb']
            orb2 = start_state['hole2_orb']
            xe, ye, ze = start_state['e_coord']
            x1, y1, z1 = start_state['hole1_coord']
            x2, y2, z2 = start_state['hole2_coord']

            # hole 1 hops: some d-orbitals might have no tpd
            if orb1 in tpd_orbs:
                for dir_ in tpd_nn_hop_dir[orb1]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1 == ['NotOnSublattice']:
                        continue

                    if not vs.check_in_vs_condition1(xe,ye,x1+vx,y1+vy,x2,y2):
                        continue
                        
                    # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                    for o1 in orbs1:
                        if o1 not in tpd_orbs:
                            continue
                        # consider Pauli principle
                        if not vs.check_Pauli(s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb1, dir_, o1])
                        if o12 in tpd_keys:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)
                            jj = VS.get_index(new_state)
                            #print 'tpd between ', i, jj

            # hole 2 hops; some d-orbitals might have no tpd
            if orb2 in tpd_orbs:
                for dir_ in tpd_nn_hop_dir[orb2]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                    if orbs2 == ['NotOnSublattice']:
                        continue

                    if not vs.check_in_vs_condition1(xe,ye,x1,y1,x2+vx,y2+vy):
                        continue
                        
                    for o2 in orbs2:
                        if o2 not in tpd_orbs:
                            continue
                        # consider Pauli principle
                        if not vs.check_Pauli(s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb2, dir_, o2])
                        if o12 in tpd_keys:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)
                            jj = VS.get_index(new_state)
                            #print 'tpd between ', i, jj
                            
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print 'check tpd hopping matrix = ', out
    #util.check_dense_matrix_hermitian(out)
    
    return out

def create_tNiOs_nn_matrix_from_no_eh(VS, tNiOs_nn_hop_dir, tNiOs_orbs, tNiOs_nn_hop_fac):
    '''
    set hop from no_eh to one_eh state
    
    The only difference that leads to different results from create_tNiOs_nn_matrix_from_one_eh
    is that pam.tNiOs*ph, when create a one-eh state there can be a ph factor
    
    Create nearest neighbor (NN) Ni to O vacancy hopping part of the Hamiltonian
    Both d10L and d9 excite eh pair
    Only dz2 electron can hop to S, dz2 hole cannot since S is full of holes
    see George's email on Jun.24, 2021
    
    For hopping signs, see George's email on Aug.19, 2021:
    dx2-y2 hoping to the O px in the positive x direction should be positive for electrons and for O in the minus x
    directions should be negative for electrons, i.e. the hoping integral should be minus sign of overlap integral
    between two neighboring atoms.
    Os-Os hoping in plane should have the same sign as Os-Os perpendicular to the plane and both should be negative 
    for electrons (note for Os we use electron language)
    So for electrons the the hoping between two s orbitals on different atoms is negative in the Hamiltonian 
    Simply think of -2t*cos(kx), which is parabola opening up so we must have that - sign
    
    Note that d3z2r2 has +,-,+ sign structure so that it is - in x-y plane and dz2-Os hopping integral is positive
    so that for electrons the hopping should be negative
    '''    
    print ("start create_tNiOs_nn_matrix (from no_eh to one_eh)")
    print ("===================================================")
    
    dim = VS.dim
    data = []
    row = []
    col = []
    nhop = 0
    
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # set the transition matrix from no_eh to one_eh
        # then directly set the transpose matrix element
        if start_state['type'] == 'one_hole_no_eh':
            s1 = start_state['hole_spin']
            orb1 = start_state['hole_orb']
            x1, y1, z1 = start_state['hole_coord']
            
            # determine possible e spin:
            # d9 can be dz2 so that the other dz2 hole must have opposite spin
            # so new hole should be the oppo of original dz2 hole spin
            if orb1=='d3z2r2':
                s2 = util.oppo_spin(s1)
                if pam.eh_spin_def=='same':
                    ses = [s2] 
                elif pam.eh_spin_def=='oppo':
                    ses = [s1] 
            else:
                ses = ['up','dn']
                
            for uz in [-1,1]:
                for se in ses:  
                    if pam.eh_spin_def=='same':
                        s2 = se  
                    elif pam.eh_spin_def=='oppo':
                        s2 = util.oppo_spin(se)
                       
                    ###################################
                    tmp_state = vs.create_one_hole_one_eh_state(se,'Os',0,0,uz,s1,orb1,x1,y1,z1,s2,'d3z2r2',0,0,0)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    set_matrix_element(row,col,data,new_state,i,VS,-pam.tNiOs*ph)

                    tse = new_state['e_spin']
                    ts1 = new_state['hole1_spin']
                    ts2 = new_state['hole2_spin']
                    torbe = new_state['e_orb']
                    torb1 = new_state['hole1_orb']
                    torb2 = new_state['hole2_orb']
                    txe, tye, tze = new_state['e_coord']
                    tx1, ty1, tz1 = new_state['hole1_coord']
                    tx2, ty2, tz2 = new_state['hole2_coord']

                    #print 'new state: ', se,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3

                    # set transpose element from no_eh to one_eh
                    row_index = VS.get_index(new_state)
                    if row_index != None:
                        data.append(-pam.tNiOs*ph)
                        row.append(i)
                        col.append(row_index)
                        #print 'tNiOs between ', i, row_index, pam.tNiOs*ph
                        nhop += 1
                        
                    ###################################
                    # new excited hole as 1st hole of new excited state (do not forget this state !!!)
                    tmp_state = vs.create_one_hole_one_eh_state(se,'Os',0,0,uz,s2,'d3z2r2',0,0,0,s1,orb1,x1,y1,z1)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    
                    # avoid double counting
                    if VS.get_index(new_state)!=row_index:
                        print ('Need more hoppings')
                        set_matrix_element(row,col,data,new_state,i,VS,-pam.tNiOs*ph)

                        # set transpose element from no_eh to one_eh
                        row_index = VS.get_index(new_state)
                        if row_index != None:
                            data.append(-pam.tNiOs*ph)
                            row.append(i)
                            col.append(row_index)
                            #print 'tNiOs between 1st hole', i, row_index
                            nhop += 1

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    #print 'Total No. of tNiOs', nhop
    #print 'tNiOs between '
    #print row
    #print col
    
    #idx = list(col).index(None)
    #print idx, row[idx],col[idx],data[idx]
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print 'check Ni-Os hopping matrix = ', out
    #util.check_dense_matrix_hermitian(out)
    
    return out

def create_tNiOs_nn_matrix_from_one_eh(VS, tNiOs_nn_hop_dir, tNiOs_orbs, tNiOs_nn_hop_fac):
    '''
    NOTE:
    The GS using this routine seems not correct because the two spins are not degenerate !!!
    Probably because of the phase of the state
    compare it with the above no_eh version routine
    
    set hop from one_eh to no_eh state
    
    Note (Jul.18, 2021):
    This routine has subtle bugs because of vs.make_state_canonical
    Suppose two no-eh state 1: a1_up; 2: a1_dn
    They both can excite to 3: e_dn, a1_up, a1_dn
    Now if starting from one-eh state and hop to no-eh state as in this routine,
    3 can only hop to state 2 but not state 1 
    So we have to implement starting from no-eh state to one-eh state !!!
    See create_tNiOs_nn_matrix above
    '''    
    print ("start create_tNiOs_nn_matrix (from one_eh to no_eh)")
    print ("===================================================")
    
    dim = VS.dim
    tNiOs_keys = tNiOs_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    nhop = 0
    
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # set the transition matrix from one_eh to no_eh
        # then directly set the transpose matrix element
        if start_state['type'] == 'one_hole_one_eh':
            se = start_state['e_spin']
            orbe = start_state['e_orb']
            xe, ye, ze = start_state['e_coord']                
            s1 = start_state['hole1_spin']
            s2 = start_state['hole2_spin']
            orb1 = start_state['hole1_orb']
            orb2 = start_state['hole2_orb']
            x1, y1, z1 = start_state['hole1_coord']
            x2, y2, z2 = start_state['hole2_coord']
            
            if orbe!='Os':
                continue
                
            # only two locations of Os can hop e to Ni 
            if not (xe==ye==0):
                continue
                
            # excited e-h state at least one hole should be on Ni
            if (orb1!='d3z2r2') and (orb2!='d3z2r2'):
                continue
            
            # hole1
            if orb1=='d3z2r2':
                # note that be careful with the usage of continue, which will switch to next for loop index
                # so that skip everything after continue
                if pam.eh_spin_def=='oppo' and s1!=se:
                    #print 'hole 1 tNiOs state', i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2
                    vac_state = vs.create_one_hole_no_eh_state(s2,orb2,x2,y2,z2)
                    set_matrix_element(row,col,data,vac_state,i,VS,-pam.tNiOs)

                    # set transpose element from no_eh to one_eh
                    row_index = VS.get_index(vac_state)
                    if row_index != None:
                        data.append(-pam.tNiOs)
                        row.append(i)
                        col.append(row_index)
                        #print 'tNiOs between ', row_index, i
                        nhop += 1
                        
                elif pam.eh_spin_def=='same' and s1==se:
                    vac_state = vs.create_one_hole_no_eh_state(s2,orb2,x2,y2,z2)
                    set_matrix_element(row,col,data,vac_state,i,VS,-pam.tNiOs)

                    # set transpose element from no_eh to one_eh
                    row_index = VS.get_index(vac_state)
                    if row_index != None:
                        data.append(-pam.tNiOs)
                        row.append(i)
                        col.append(row_index)
                        #print 'tNiOs between ', row_index, i
                        nhop += 1

            # hole2
            if orb2=='d3z2r2':
                if pam.eh_spin_def=='oppo' and s2!=se:
                    #print 'hole 1 tNiOs state', i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2
                    vac_state = vs.create_one_hole_no_eh_state(s1,orb1,x1,y1,z1)
                    set_matrix_element(row,col,data,vac_state,i,VS,-pam.tNiOs)

                    # set transpose element from no_eh to one_eh
                    row_index = VS.get_index(vac_state)
                    if row_index != None:
                        data.append(-pam.tNiOs)
                        row.append(i)
                        col.append(row_index)
                        #print 'tNiOs between ', row_index, i
                        nhop += 1
                        
                elif pam.eh_spin_def=='same' and s2==se:
                    vac_state = vs.create_one_hole_no_eh_state(s1,orb1,x1,y1,z1)
                    set_matrix_element(row,col,data,vac_state,i,VS,-pam.tNiOs)

                    # set transpose element from no_eh to one_eh
                    row_index = VS.get_index(vac_state)
                    if row_index != None:
                        data.append(-pam.tNiOs)
                        row.append(i)
                        col.append(row_index)
                        #print 'tNiOs between ', row_index, i
                        nhop += 1
                     
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    print ('Total No. of tNiOs', nhop)
    #print 'tNiOs between '
    #print row
    #print col
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print 'check Ni-Os hopping matrix = ', out
    #util.check_dense_matrix_hermitian(out)
    
    return out

def create_tpp_nn_matrix(VS,tpp_nn_hop_fac): 
    '''
    similar to comments in create_tpp_nn_matrix
    '''   
    print ("start create_tpp_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpp_orbs = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # only hole hops with tpd
        if start_state['type'] == 'one_hole_no_eh':
            s1 = start_state['hole_spin']
            orb1 = start_state['hole_orb']
            x1, y1, z1 = start_state['hole_coord']
            
            # hole1 hops: only p-orbitals has t_pp 
            if orb1 in pam.O_orbs: 
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    
                    if orbs1!=pam.O1_orbs and orbs1!=pam.O2_orbs:
                        continue
                        
                    # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                    for o1 in orbs1:
                        tmp_state = vs.create_one_hole_no_eh_state(s1,o1,x1+vx,y1+vy,z1+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)
                        
                        o12 = sorted([orb1, dir_, o1])
                        o12 = tuple(o12)
                        if o12 in tpp_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

        if start_state['type'] == 'one_hole_one_eh':
            se = start_state['e_spin']
            s1 = start_state['hole1_spin']
            s2 = start_state['hole2_spin']
            orbe = start_state['e_orb']
            orb1 = start_state['hole1_orb']
            orb2 = start_state['hole2_orb']
            xe, ye, ze = start_state['e_coord']
            x1, y1, z1 = start_state['hole1_coord']
            x2, y2, z2 = start_state['hole2_coord']

            # hole1 hops: only p-orbitals has t_pp 
            if orb1 in pam.O_orbs: 
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1!=pam.O1_orbs and orbs1!=pam.O2_orbs:
                        continue

                    if not vs.check_in_vs_condition1(xe,ye,x1+vx,y1+vy,x2,y2):
                        continue
                        
                    # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                    for o1 in orbs1:
                        # consider Pauli principle
                        if not vs.check_Pauli(s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb1, dir_, o1])
                        o12 = tuple(o12)
                        if o12 in tpp_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

            # hole2 hops: only p-orbitals has t_pp 
            if orb2 in pam.O_orbs: 
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                    if orbs2!=pam.O1_orbs and orbs2!=pam.O2_orbs:
                        continue

                    if not vs.check_in_vs_condition1(xe,ye,x1,y1,x2+vx,y2+vy):
                        continue
                        
                    # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                    for o2 in orbs2:
                        # consider Pauli principle
                        if not vs.check_Pauli(s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb2, dir_, o2])
                        o12 = tuple(o12)
                        if o12 in tpp_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print 'check tpp hopping matrix = ', out
    #util.check_dense_matrix_hermitian(out)

    return out

def create_tOsOs_nn_matrix(VS,tOsOs,tOsOs_p): 
    '''
    similar to comments in create_tOsOs_nn_matrix
    
    For hopping signs, see George's email on Aug.19, 2021:
    dx2-y2 hoping to the O px in the positive x direction should be positive for electrons and for O in the minus x
    directions should be negative for electrons, i.e. the hoping integral should be minus sign of overlap integral
    between two neighboring atoms.
    Os-Os hoping in plane should have the same sign as Os-Os perpendicular to the plane and both should be negative 
    for electrons (note for Os we use electron language)
    So for electrons the the hoping between two s orbitals on different atoms is negative in the Hamiltonian 
    Simply think of -2t*cos(kx), which is parabola opening up so we must have that - sign
    '''   
    print ("start create_tOsOs_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        if start_state['type'] == 'one_hole_one_eh':
            se = start_state['e_spin']
            s1 = start_state['hole1_spin']
            s2 = start_state['hole2_spin']
            orbe = start_state['e_orb']
            orb1 = start_state['hole1_orb']
            orb2 = start_state['hole2_orb']
            xe, ye, ze = start_state['e_coord']
            x1, y1, z1 = start_state['hole1_coord']
            x2, y2, z2 = start_state['hole2_coord']

            # Os electron hops 
            for dir_ in tOs_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbse = lat.get_unit_cell_rep(xe+vx, ye+vy, ze+vz)
                if orbse != pam.Ovacancy_orbs:
                    continue

                if not vs.check_in_vs_condition1(xe+vx,ye+vy,x1,y1,x2,y2):
                    continue
                  
                for oe in orbse:
                    tmp_state = vs.create_one_hole_one_eh_state(se,oe,xe+vx,ye+vy,ze+vz,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    if dir_ in ['T2','B2']:
                        set_matrix_element(row,col,data,new_state,i,VS,-tOsOs_p*ph)
                    else:
                        set_matrix_element(row,col,data,new_state,i,VS,-tOsOs*ph)
                        
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    #print 'check Os-Os hopping matrix = ', out
    #util.check_dense_matrix_hermitian(out)

    return out

def create_edepeOs_diag_matrix(VS,A,ep):
    '''
    Create diagonal part of the site energies. 
    For two hole problem here, the reference energy is e_d9=0
    d8's original energy ~A should reduce to A/2, which is completed in setting interaction matrix
    and d10 should be have that remaining U/2 or A/2 energy
    see George's email on Jun.21, 2021
    '''    
    print ("start create_edepeOs_diag_matrix")
    print ("================================")
    dim = VS.dim
    data = []
    row = []
    col = []
    idxs = np.zeros(dim)

    for i in range(0,dim):
        diag_el = 0.
        state = VS.get_state(VS.lookup_tbl[i])

        if state['type'] == 'one_hole_no_eh':
            orb = state['hole_orb']

            # d9
            if orb in pam.Ni_orbs:
                diag_el += pam.ed[orb] 
            # d10L
            elif orb in pam.O_orbs: 
                diag_el += ep + A/2.
            data.append(diag_el); row.append(i); col.append(i)
            
        elif state['type'] == 'one_hole_one_eh':
            orbe = state['e_orb']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']

            # d9Ls
            if orb1 in pam.Ni_orbs and orb2 in pam.O_orbs:
                diag_el += pam.ed[orb1] + ep + pam.eOs
            elif orb2 in pam.Ni_orbs and orb1 in pam.O_orbs: 
                diag_el += pam.ed[orb2] + ep + pam.eOs
                
            # d8s
            elif orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs: 
                diag_el += pam.ed[orb1] + pam.ed[orb2] + pam.eOs 
                
            # d10L2s
            elif orb1 in pam.O_orbs and orb2 in pam.O_orbs: 
                diag_el += 2.*ep  + pam.eOs + A/2.  # A/2 see above

            data.append(diag_el); row.append(i); col.append(i)
            
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    #print min(data)
    #print len(row), len(col)
    
   # for ii in range(0,len(row)):
   #     if data[ii]==0:
   #         print ii
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out
    
def get_double_occu_list(VS):
    '''
    Get the list of states that two holes are both d or p-orbitals
    '''
    dim = VS.dim
    d_list = []
    p_list = []
    
    for i in range(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
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
            
            if (x1,y1) == (x2,y2):
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
                    d_list.append(i)
                    #print "d_double: one_eh", i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2,s3,orb3,x3,y3
                elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
                    p_list.append(i)
                    #print "p_double: ", s1,orb1,x1,y1,s2,orb2,x2,y2
                    
    print ("len(d_list)", len(d_list))
    print ("len(p_list)", len(p_list))
    
    return d_list, p_list

def create_interaction_matrix_ALL_syms(VS,d_double,p_double,S_val, Sz_val, AorB_sym, A, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets including all symmetries
    
    Loop over all d_double states, find the corresponding sym channel; 
    the other loop over all d_double states, if it has same sym channel and S, Sz
    enter into the matrix element
    
    There are some complications or constraints due to three holes and one Nd electron:
    From H_matrix_reducing_VS file, to set up interaction between states i and j:
    1. i and j belong to the same type, same order of orbitals to label the state (idxi==idxj below)
    2. i and j's spins are same; or L and s should also have same spin
    3. Positions of L and Nd-electron should also be the same
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []
    
    channels = ['1A1','1A2','3A2','1B1','3B1','1E','3E','1B2','3B2']

    for sym in channels:
        state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(A, sym)
        sym_orbs = state_order.keys()
        #print "orbitals in sym ", sym, "= ", sym_orbs

        for i in d_double:
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            itype = state['type']
            
            assert(itype == 'one_hole_one_eh')
            
            ise = state['e_spin']
            is1 = state['hole1_spin']
            is2 = state['hole2_spin']
            ioe = state['e_orb']
            io1 = state['hole1_orb']
            io2 = state['hole2_orb']
            ixe, iye, ize = state['e_coord']
            ix1, iy1, iz1 = state['hole1_coord']
            ix2, iy2, iz2 = state['hole2_coord']

            iepos=[ixe, iye, ize]

            o12 = sorted([io1,io2])
            o12 = tuple(o12)

            # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
            S12  = S_val[i]
            Sz12 = Sz_val[i]

            # continue only if (o1,o2) is within desired sym
            if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
                continue

            if (io1==io2=='dxz' or io1==io2=='dyz') and AorB_sym[i]!=AorB:
                continue

            # get the corresponding index in sym for setting up matrix element
            idx1 = state_order[o12]

            for j in d_double:
                if j<i:
                    continue

                state = VS.get_state(VS.lookup_tbl[j])
                jtype = state['type']

                if jtype!=itype:
                    continue

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

                if not (jse==ise and joe==ioe and jepos==iepos):
                    continue

                o34 = sorted([jo1,jo2])
                o34 = tuple(o34)
                S34  = S_val[j]
                Sz34 = Sz_val[j]

                if (jo1==jo2=='dxz' or jo1==jo2=='dyz') and AorB_sym[j]!=AorB:
                    continue

                # only same total spin S and Sz state have nonzero matrix element
                if o34 in sym_orbs and S34==S12 and Sz34==Sz12:
                    idx2 = state_order[o34]

                    #print o12[0],o12[1],S12,Sz12," ",o34[0],o34[1],S34,Sz34," ", interaction_mat[idx1][idx2]
                    #print idx1, idx2

                    val = interaction_mat[idx1][idx2]
                    data.append(val); row.append(i); col.append(j)

                    # make symmetric interaction matrix
                    if j!=i:
                        data.append(val); row.append(j); col.append(i)

    # Create Upp matrix for p-orbital multiplets
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out
