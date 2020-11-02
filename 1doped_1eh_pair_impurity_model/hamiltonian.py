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
                      'L2': (-2,0,0),\
                      'R2': (2,0,0),\
                      'U2': (0,2,0),\
                      'D2': (0,-2,0),\
                      'pzL': (-1,0,1),\
                      'pzR': (1,0,1),\
                      'pzU': (0,1,1),\
                      'pzD': (0,-1,1),\
                      'mzL': (-1,0,-1),\
                      'mzR': (1,0,-1),\
                      'mzU': (0,1,-1),\
                      'mzD': (0,-1,-1)}
tpp_nn_hop_dir = ['UR','UL','DL','DR']
tNdNd_nn_hop_dir = ['U2','D2','R2','L2']

def set_tpd_tpp_tNiNd(Norb,tpd,tpp,tNiNd,pds,pdp,pps,ppp):
    # dxz and dyz has no tpd hopping
    if pam.Norb==3:
        tpd_nn_hop_dir = {'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==8:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
        tNiNd_nn_hop_dir = {'d3z2r2' : ['UR','UL','DL','DR'],\
                            'dxy'    : ['UR','UL','DL','DR'],\
                            'Nd_s'   : ['UR','UL','DL','DR']}
    elif pam.Norb==10 or pam.Norb==11:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D']}
        tNiNd_nn_hop_dir = {'d3z2r2' : ['UR','UL','DL','DR'],\
                            'dxy'    : ['UR','UL','DL','DR'],\
                            'Nd_s'   : ['UR','UL','DL','DR']}
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
        tNiNd_nn_hop_dir = {'d3z2r2' : ['UR','UL','DL','DR'],\
                            'dxy'    : ['UR','UL','DL','DR'],\
                            'Nd_s'   : ['UR','UL','DL','DR']}
    if pam.Norb==3:
        if_tpd_nn_hop = {'dx2y2' : 1,\
                         'px'    : 1,\
                         'py'    : 1}
    elif pam.Norb==8:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 0,\
                         'dxz'   : 0,\
                         'dyz'   : 0,\
                         'px'    : 1,\
                         'py'    : 1}
        if_tNiNd_nn_hop = {'d3z2r2': 1,\
                           'dx2y2' : 0,\
                           'dxy'   : 1,\
                           'dxz'   : 0,\
                           'dyz'   : 0,\
                           'Nd_s'  : 1}
    elif pam.Norb==10:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 1,\
                         'dxz'   : 0,\
                         'dyz'   : 0,\
                         'px1'   : 1,\
                         'py1'   : 1,\
                         'px2'   : 1,\
                         'py2'   : 1}
        if_tNiNd_nn_hop = {'d3z2r2': 1,\
                           'dx2y2' : 0,\
                           'dxy'   : 1,\
                           'dxz'   : 0,\
                           'dyz'   : 0,\
                           'Nd_s'  : 1}
    elif pam.Norb==11:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 1,\
                         'dxz'   : 0,\
                         'dyz'   : 0,\
                         'apz'   : 0,\
                         'px1'   : 1,\
                         'py1'   : 1,\
                         'px2'   : 1,\
                         'py2'   : 1}
        if_tNiNd_nn_hop = {'d3z2r2': 1,\
                           'dx2y2' : 0,\
                           'dxy'   : 1,\
                           'dxz'   : 0,\
                           'dyz'   : 0,\
                           'Nd_s'  : 1}
    elif pam.Norb==12:
        if_tpd_nn_hop = {'d3z2r2': 1,\
                         'dx2y2' : 1,\
                         'dxy'   : 1,\
                         'dxz'   : 1,\
                         'dyz'   : 1,\
                         'px1'   : 1,\
                         'py1'   : 1,\
                         'pz1'   : 1,\
                         'px2'   : 1,\
                         'py2'   : 1,\
                         'pz2'   : 1}
        if_tNiNd_nn_hop = {'d3z2r2': 1,\
                           'dx2y2' : 0,\
                           'dxy'   : 1,\
                           'dxz'   : 0,\
                           'dyz'   : 0,\
                           'Nd_s'  : 1}
    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    if pam.Norb==3:
        tpd_nn_hop_fac = {('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
    elif pam.Norb==8:
        # d3z2r2 has -,+,- sign structure so that it is positive in x-y plane
        # google results shows +,-,+ sign, but that induces abnormal spectra
        tpd_nn_hop_fac = {('d3z2r2','L','px'):  tpd/np.sqrt(3),\
                          ('d3z2r2','R','px'): -tpd/np.sqrt(3),\
                          ('d3z2r2','U','py'): -tpd/np.sqrt(3),\
                          ('d3z2r2','D','py'):  tpd/np.sqrt(3),\
                          ('dx2y2','L','px'):   tpd,\
                          ('dx2y2','R','px'):  -tpd,\
                          ('dx2y2','U','py'):   tpd,\
                          ('dx2y2','D','py'):  -tpd,\
                          # below just inverse dir of the above one by one
                          ('px','R','d3z2r2'):  tpd/np.sqrt(3),\
                          ('px','L','d3z2r2'): -tpd/np.sqrt(3),\
                          ('py','D','d3z2r2'): -tpd/np.sqrt(3),\
                          ('py','U','d3z2r2'):  tpd/np.sqrt(3),\
                          ('px','R','dx2y2'):   tpd,\
                          ('px','L','dx2y2'):  -tpd,\
                          ('py','D','dx2y2'):   tpd,\
                          ('py','U','dx2y2'):  -tpd}
        # only dxy and d3z2r2 hops to Nd s-like orbital (note that here use electron language!!!)
        tNiNd_nn_hop_fac = {('Nd_s','UR','dxy'): -tNiNd*np.sqrt(3),\
                            ('Nd_s','UL','dxy'):  tNiNd*np.sqrt(3),\
                            ('DL','Nd_s','dxy'): -tNiNd*np.sqrt(3),\
                            ('DR','Nd_s','dxy'):  tNiNd*np.sqrt(3),\
                            ('Nd_s','UR','d3z2r2'): -tNiNd,\
                            ('Nd_s','UL','d3z2r2'): -tNiNd,\
                            ('DL','Nd_s','d3z2r2'): -tNiNd,\
                            ('DR','Nd_s','d3z2r2'): -tNiNd}
    elif pam.Norb==10 or pam.Norb==11:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'):  pds/2.0,\
                          ('d3z2r2','R','px1'): -pds/2.0,\
                          ('d3z2r2','U','py2'): -pds/2.0,\
                          ('d3z2r2','D','py2'):  pds/2.0,\
                          ('dx2y2','L','px1'):   pds*c,\
                          ('dx2y2','R','px1'):  -pds*c,\
                          ('dx2y2','U','py2'):   pds*c,\
                          ('dx2y2','D','py2'):  -pds*c,\
                          ('dxy','L','py1'):  -pdp,\
                          ('dxy','R','py1'):   pdp,\
                          ('dxy','U','px2'):   pdp,\
                          ('dxy','D','px2'):  -pdp,\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'):  pds/2.0,\
                          ('px1','L','d3z2r2'): -pds/2.0,\
                          ('py2','D','d3z2r2'): -pds/2.0,\
                          ('py2','U','d3z2r2'):  pds/2.0,\
                          ('px1','R','dx2y2'):   pds*c,\
                          ('px1','L','dx2y2'):  -pds*c,\
                          ('py2','D','dx2y2'):   pds*c,\
                          ('py2','U','dx2y2'):  -pds*c,\
                          ('py1','R','dxy'):  -pdp,\
                          ('py1','L','dxy'):   pdp,\
                          ('px2','D','dxy'):   pdp,\
                          ('px2','U','dxy'):  -pdp}
        # only dxy and d3z2r2 hops to Nd s-like orbital (note that here use electron language!!!)
        tNiNd_nn_hop_fac = {('Nd_s','UR','dxy'): -tNiNd*np.sqrt(3),\
                            ('Nd_s','UL','dxy'):  tNiNd*np.sqrt(3),\
                            ('DL','Nd_s','dxy'): -tNiNd*np.sqrt(3),\
                            ('DR','Nd_s','dxy'):  tNiNd*np.sqrt(3),\
                            ('Nd_s','UR','d3z2r2'): -tNiNd,\
                            ('Nd_s','UL','d3z2r2'): -tNiNd,\
                            ('DL','Nd_s','d3z2r2'): -tNiNd,\
                            ('DR','Nd_s','d3z2r2'): -tNiNd}
    elif pam.Norb==12:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'):  pds/2.0,\
                          ('d3z2r2','R','px1'): -pds/2.0,\
                          ('d3z2r2','U','py2'): -pds/2.0,\
                          ('d3z2r2','D','py2'):  pds/2.0,\
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
                          ('px1','R','d3z2r2'):  pds/2.0,\
                          ('px1','L','d3z2r2'): -pds/2.0,\
                          ('py2','D','d3z2r2'): -pds/2.0,\
                          ('py2','U','d3z2r2'):  pds/2.0,\
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
        # only dxy and d3z2r2 hops to Nd s-like orbital (note that here use electron language!!!)
        tNiNd_nn_hop_fac = {('Nd_s','UR','dxy'): -tNiNd*np.sqrt(3),\
                            ('Nd_s','UL','dxy'):  tNiNd*np.sqrt(3),\
                            ('DL','Nd_s','dxy'): -tNiNd*np.sqrt(3),\
                            ('DR','Nd_s','dxy'):  tNiNd*np.sqrt(3),\
                            ('Nd_s','UR','d3z2r2'): -tNiNd,\
                            ('Nd_s','UL','d3z2r2'): -tNiNd,\
                            ('DL','Nd_s','d3z2r2'): -tNiNd,\
                            ('DR','Nd_s','d3z2r2'): -tNiNd}
    ########################## tpp below ##############################
    if pam.Norb==3 or pam.Norb==8:
        tpp_nn_hop_fac = {('UR','px','py'): -tpp,\
                          ('UL','px','py'):  tpp,\
                          ('DL','px','py'): -tpp,\
                          ('DR','px','py'):  tpp}
    elif pam.Norb==10 or pam.Norb==11 or pam.Norb==12:
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
    ########################## apical pz below ##############################
    if pam.Norb==11:
        if_apz2p_hop = {'d3z2r2': 0,\
                        'dx2y2' : 0,\
                        'dxy'   : 0,\
                        'dxz'   : 0,\
                        'dyz'   : 0,\
                        'apz'   : 1,\
                        'px1'   : 1,\
                        'py1'   : 0,\
                        'px2'   : 0,\
                        'py2'   : 1}
        apz2p_hop_dir = {'apz': ['mzL','mzR','mzU','mzD'],\
                         'px1': ['pzL','pzR'],\
                         'py2': ['pzU','pzD']}
        apz2p_hop_fac = {('apz','mzL','px1'): -0.25*(ppp+pps),\
                         ('apz','mzR','px1'):  0.25*(ppp+pps),\
                         ('apz','mzU','py2'):  0.25*(ppp+pps),\
                         ('apz','mzD','py2'): -0.25*(ppp+pps),\
                         # below just inverse dir of the above one by one
                         ('px1','pzR','apz'): -0.25*(ppp+pps),\
                         ('px1','pzL','apz'):  0.25*(ppp+pps),\
                         ('py2','pzD','apz'):  0.25*(ppp+pps),\
                         ('py2','pzU','apz'): -0.25*(ppp+pps)}
    else:
        apz2p_hop_dir = []
        if_apz2p_hop  = []
        apz2p_hop_fac = []
        
    return tpd_nn_hop_dir, if_tpd_nn_hop, tpd_nn_hop_fac, \
           tNiNd_nn_hop_dir, if_tNiNd_nn_hop, tNiNd_nn_hop_fac, \
           tpp_nn_hop_fac, apz2p_hop_dir, if_apz2p_hop, apz2p_hop_fac

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
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,           B+C,           B+C,       0], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             3.*B+C,        3.*B+C,       0], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   3.*B+C,        3.*B+C,       0], \
                           [B+C,          3.*B+C,       3.*B+C,        A+4.*B+3.*C,   3.*B+C,       B*fac], \
                           [B+C,          3.*B+C,       3.*B+C,        3.*B+C,        A+4.*B+3.*C, -B*fac], \
                           [0,            0,            0,              B*fac,         -B*fac,      A+2.*C]]
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
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,        fac*(B+C)], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             fac*(3.*B+C)], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   fac*(3.*B+C)], \
                           [fac*(B+C),    fac*(3.*B+C), fac*(3.*B+C),  A+7.*B+4.*C]]
    if sym=='1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dx2y2'): 0,\
                       ('dxz','dxz')     : 1,\
                       ('dyz','dyz')     : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
    if sym=='1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0}
        interaction_mat = [[A+4.*B+2.*C]]
    if sym=='3A2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0,\
                       ('dxz','dyz')  : 1}
        interaction_mat = [[A+4.*B,   6.*B], \
                           [6.*B,     A-5.*B]]
    if sym=='3B1':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('d3z2r2','dx2y2'): 0}
        interaction_mat = [[A-8.*B]]
    if sym=='1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxy'): 0,\
                       ('dxz','dyz')   : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
    if sym=='3B2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = -1
        state_order = {('d3z2r2','dxy'): 0}
        interaction_mat = [[A-8.*B]]
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
        interaction_mat = [[A+3.*B+2.*C,  0,           -B*fac,      0,          0,        -B*fac], \
                           [0,            A+3.*B+2.*C,  0,          B*fac,     -B*fac,     0], \
                           [-B*fac,       0,            A+B+2.*C,   0,          0,        -3.*B], \
                           [0,            B*fac,        0,          A+B+2.*C,   3.*B,      0 ], \
                           [0,           -B*fac,        0,          3.*B,       A+B+2.*C,  0], \
                           [-B*fac,       0,           -3.*B,       0,          0,         A+B+2.*C]]
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
        interaction_mat = [[A+B,         0,         -3.*B*fac,    0,          0,        -3.*B*fac], \
                           [0,           A+B,        0,           3.*B*fac,  -3.*B*fac,  0], \
                           [-3.*B*fac,   0,          A-5.*B,      0,          0,         3.*B], \
                           [0,           3.*B*fac,   0,           A-5.*B,    -3.*B,      0 ], \
                           [0,          -3.*B*fac,   0,          -3.*B,       A-5.*B,    0], \
                           [-3.*B*fac,   0,          3.*B,        0,          0,         A-5.*B]]
        
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

def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, if_tpd_nn_hop, tpd_nn_hop_fac):
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
    print "start create_tpd_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    tpd_orbs = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in xrange(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        
        # only hole hops with tpd
        if start_state['type'] == 'one_hole_no_eh':
            s1 = start_state['spin']
            orb1 = start_state['hole_orb']
            x1,y1,z1 = start_state['hole_coord']

            # hole hops; some d-orbitals might have no tpd
            if if_tpd_nn_hop[orb1] == 1:
                for dir_ in tpd_nn_hop_dir[orb1]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1 == ['NotOnSublattice'] or orbs1==pam.Nd_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1+vx,y1+vy,0,0):
                        continue
                        
                    # consider t_pd for all cases
                    for o1 in orbs1:
                        if if_tpd_nn_hop[o1] == 0:
                            continue

                        tmp_state = vs.create_one_hole_no_eh_state(s1,o1,x1+vx,y1+vy,z1+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb1, dir_, o1])
                        if o12 in tpd_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)
          
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

            # hole 1 hops: some d-orbitals might have no tpd
            if if_tpd_nn_hop[orb1] == 1:
                for dir_ in tpd_nn_hop_dir[orb1]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1 == ['NotOnSublattice'] or orbs1==pam.Nd_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1+vx,y1+vy,x2,y2):
                        continue
                        
                    # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                    for o1 in orbs1:
                        if if_tpd_nn_hop[o1] == 0:
                            continue
                        # consider Pauli principle
                        if s1==s2 and o1==orb2 and (x1+vx,y1+vy,z1+vz)==(x2,y2,z2):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze, \
                                                                    s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb1, dir_, o1])
                        if o12 in tpd_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

            # hole 2 hops; some d-orbitals might have no tpd
            if if_tpd_nn_hop[orb2] == 1:
                for dir_ in tpd_nn_hop_dir[orb2]:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                    if orbs2 == ['NotOnSublattice'] or orbs2==pam.Nd_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1,y1,x2+vx,y2+vy):
                        continue
                        
                    for o2 in orbs2:
                        if if_tpd_nn_hop[o2] == 0:
                            continue
                        # consider Pauli principle
                        if s1==s2 and orb1==o2 and (x1,y1,z1)==(x2+vx, y2+vy, z2+vz):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze, \
                                                                    s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = tuple([orb2, dir_, o2])
                        if o12 in tpd_orbs:
                            set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out

def create_tNiNd_nn_matrix(VS, tNiNd_nn_hop_dir, if_tNiNd_nn_hop, tNiNd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) Ni-Nd hopping part of the Hamiltonian
    Involve both no_eh and one_eh states
    Because only allow one electron hole excitation at most, tNiNd must be 
    creation/annihilation of e-h excitation so that e can only be on
    sites nn to Ni
    By Pauli principle, the only process is e hops from Nd to Ni
    
    George said to only keep d10 to d9s hopping; so only 
    d9_dn L_up s_up and d9_up L_up s_dn connect to L_up have finite tNiNd
    '''    
    print "start create_tNiNd_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    tNiNd_orbs = tNiNd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    
    for i in xrange(0,dim):
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
            
            if if_tNiNd_nn_hop[orbe]==0:
                continue
                
            # only Nd nn to Ni can hop el to Ni with tNiNd
            if vs.calc_manhattan_dist(xe,ye,0,0)>2.1:
                continue
                
            # d8 and d10L2 are not allowed, see above George's idea
            if (orb1 in pam.Ni_orbs) and (orb2 in pam.Ni_orbs):
                continue
            if (orb1 in pam.O_orbs) and (orb2 in pam.O_orbs):
                continue
            
            # hole1
            if orb1 in pam.Ni_orbs:
                # eh annilation must be opposite spin
                if s1==se:
                    continue
                    
                if if_tNiNd_nn_hop[orb1] == 1:
                    for dir_ in tNiNd_nn_hop_dir[orbe]:
                        vx, vy, vz = directions_to_vecs[dir_]
                        orbse = lat.get_unit_cell_rep(xe+vx, ye+vy, ze+vz)
                        
                        if orbse!=pam.Ni_orbs:
                            continue
                            
                        o12 = sorted([orbe, dir_, orb1])
                        o12 = tuple(o12)
                        
                        if o12 in tNiNd_orbs:
                            #print 'hole 1 tNiNd state', i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2
                            vac_state = vs.create_one_hole_no_eh_state(s2,orb2,x2,y2,z2)
                            set_matrix_element(row,col,data,vac_state,i,VS,tNiNd_nn_hop_fac[o12])
                            
                            # set transpose element from no_eh to one_eh
                            row_index = VS.get_index(vac_state)
                            data.append(tNiNd_nn_hop_fac[o12])
                            row.append(i)
                            col.append(row_index)

            # hole2
            if orb2 in pam.Ni_orbs:
                # eh annilation must be opposite spin
                if s2==se:
                    continue
                    
                if if_tNiNd_nn_hop[orb2] == 1:
                    for dir_ in tNiNd_nn_hop_dir[orbe]:
                        vx, vy, vz = directions_to_vecs[dir_]
                        orbse = lat.get_unit_cell_rep(xe+vx, ye+vy, ze+vz)
                        
                        if orbse!=pam.Ni_orbs:
                            continue
                            
                        o12 = sorted([orbe, dir_, orb2])
                        o12 = tuple(o12)
                        
                        if o12 in tNiNd_orbs:
                            #print 'hole 2 tNiNd state', i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2
                            vac_state = vs.create_one_hole_no_eh_state(s1,orb1,x1,y1,z1)
                            set_matrix_element(row,col,data,vac_state,i,VS,tNiNd_nn_hop_fac[o12])
                            
                            # set transpose element from no_eh to one_eh
                            row_index = VS.get_index(vac_state)
                            data.append(tNiNd_nn_hop_fac[o12])
                            row.append(i)
                            col.append(row_index)
                            
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out

def create_tpp_nn_matrix(VS,tpp_nn_hop_fac): 
    '''
    similar to comments in create_tpp_nn_matrix
    '''   
    print "start create_tpp_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    data = []
    row = []
    col = []
    for i in xrange(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # only hole hops with tpd
        if start_state['type'] == 'one_hole_no_eh':
            s1 = start_state['spin']
            orb1 = start_state['hole_orb']
            x1,y1,z1 = start_state['hole_coord']
            
            # hole hops, only p-orbitals has t_pp 
            if orb1 in pam.O_orbs and orb1 not in pam.ap_orbs and orb1!='pz1' and orb1!='pz2':
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)

                    if orbs1!=pam.O_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1+vx,y1+vy,0,0): 
                        continue
                        
                    for o1 in orbs1:
                        if o1=='pz1' or o1=='pz2':
                            continue
      
                        new_state = vs.create_one_hole_no_eh_state(s1,o1,x1+vx, y1+vy, z1+vz)
                        o12 = sorted([orb1, dir_, o1])
                        o12 = tuple(o12)
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12])
          
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

            # hole1 hops: only p-orbitals has t_pp 
            if orb1 in pam.O_orbs and orb1 not in pam.ap_orbs and orb1!='pz1' and orb1!='pz2': 
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                    if orbs1 != pam.O_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1+vx,y1+vy,x2,y2):
                        continue
                        
                    # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                    for o1 in orbs1:
                        if o1=='pz1' or o1=='pz2':
                            continue

                        # consider Pauli principle
                        if s1==s2 and o1==orb2 and (x1+vx,y1+vy,z1+vz)==(x2,y2,z2):
                            continue

                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze, \
                                                                    s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb1, dir_, o1])
                        o12 = tuple(o12)
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

            # hole 2 hops, only p-orbitals has t_pp 
            if orb2 in pam.O_orbs and orb2 not in pam.ap_orbs and orb2!='pz1' and orb2!='pz2':
                for dir_ in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[dir_]
                    orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)

                    if orbs2!= pam.O_orbs:
                        continue

                    if not vs.check_in_vs_condition(x1,y1,x2+vx,y2+vy): 
                        continue
                        
                    for o2 in orbs2:
                        if o2=='pz1' or o2=='pz2':
                            continue

                        # consider Pauli principle
                        if s1==s2 and orb1==o2 and (x1,y1,z1)==(x2+vx, y2+vy, z2+vz):
                            continue
  
                        tmp_state = vs.create_one_hole_one_eh_state(se,orbe,xe,ye,ze, \
                                                                    s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                        new_state,ph = vs.make_state_canonical(tmp_state)

                        o12 = sorted([orb2, dir_, o2])
                        o12 = tuple(o12)
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)
                        
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out

def create_tNdNd_nn_matrix(VS,tNdNd): 
    '''
    similar to comments in create_tNdNd_nn_matrix
    '''   
    print "start create_tNdNd_nn_matrix"
    print "=========================="
    
    dim = VS.dim
    data = []
    row = []
    col = []
    for i in xrange(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
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

            # Nd electron hops 
            for dir_ in tNdNd_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbse = lat.get_unit_cell_rep(xe+vx, ye+vy, ze+vz)
                if orbse != pam.Nd_orbs:
                    continue

                if not (vs.check_in_vs_condition(xe+vx,ye+vy,x1,y1) and vs.check_in_vs_condition(xe+vx,ye+vy,x2,y2)):
                    continue
                  
                for oe in orbse:
                    tmp_state = vs.create_one_hole_one_eh_state(se,oe,xe+vx,ye+vy,ze+vz, \
                                                                s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    set_matrix_element(row,col,data,new_state,i,VS,tNdNd*ph)
                        
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out

def create_edep_diag_matrix(VS,ep):
    '''
    Create diagonal part of the site energies. Assume ed = 0!
    '''    
    print "start create_edep_diag_matrix"
    print "============================="
    dim = VS.dim
    data = []
    row = []
    col = []

    for i in xrange(0,dim):
        diag_el = 0.
        state = VS.get_state(VS.lookup_tbl[i])

        if state['type'] == 'one_hole_no_eh':
            orb1 = state['hole_orb']

            if orb1 in pam.Ni_orbs: 
                diag_el += pam.ed[orb1]
            elif orb1 in pam.O_orbs:
                diag_el += ep

            data.append(diag_el); row.append(i); col.append(i)
            
        if state['type'] == 'one_hole_one_eh':
            orbe = state['e_orb']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']

            if orbe in pam.Nd_orbs: 
                diag_el += pam.eNd
               
            if orb1 in pam.Ni_orbs: 
                diag_el += pam.ed[orb1]
            elif orb1 in pam.O_orbs:
                diag_el += ep
                
            if orb2 in pam.Ni_orbs: 
                diag_el += pam.ed[orb2]
            elif orb2 in pam.O_orbs:
                diag_el += ep

            data.append(diag_el); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
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
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        if state['type'] == 'one_hole_one_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']

            if (x1,y1) == (x2,y2):
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
                    d_list.append(i)
                    #print "d_double: idx", i, s1,orb1,x1,y1,s2,orb2,x2,y2
                elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
                    p_list.append(i)
                elif orb1 in pam.ap_orbs and orb2 in pam.ap_orbs:
                    p_list.append(i)
                    #print "p_double: ", s1,orb1,x1,y1,s2,orb2,x2,y2

    print "len(d_list)", len(d_list)
    print "len(p_list)", len(p_list)
    
    return d_list, p_list

def create_interaction_matrix(VS,sym,d_double,p_double,S_val, Sz_val, AorB_sym, A, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets
    
    Cannot directly use the table in thesis or PRB paper since 
    those matrix elements are between singlets/triplets,
    need transform the basis in create_singlet_triplet_basis_change_matrix
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []

    # Create Coulomb-exchange matrix for d-orbital multiplets
    state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(A, sym)
    sym_orbs = state_order.keys()
    print "orbitals in sym ", sym, "= ", sym_orbs

    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        if state['type'] == 'one_hole_one_eh':
            o1 = state['hole1_orb']
            o2 = state['hole2_orb']
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

            # get the corresponding index in sym for setting up matrix element
            idx1 = state_order[o12]
            for j in d_double:
                state = VS.get_state(VS.lookup_tbl[j])
                o3 = state['hole1_orb']
                o4 = state['hole2_orb']
                o34 = sorted([o3,o4])
                o34 = tuple(o34)
                S34  = S_val[j]
                Sz34 = Sz_val[j]

                if (o3==o4=='dxz' or o3==o4=='dyz') and AorB_sym[j]!=AorB:
                    continue

                # only same total spin S and Sz state have nonzero matrix element
                if o34 in sym_orbs and S34==S12 and Sz34==Sz12:
                    idx2 = state_order[o34]

                    #print o12[0],o12[1],S12,Sz12," ",o34[0],o34[1],S34,Sz34," ", interaction_mat[idx1][idx2]
                    #print idx1, idx2

                    val = interaction_mat[idx1][idx2]
                    data.append(val); row.append(i); col.append(j)

            # get index for desired dd states
            # Note: for transformed basis of singlet/triplet
            # index can differ from that in original basis
            if 'dx2y2' in o12:
                # for triplet, only need one Sz state; other Sz states have the same A(w)
                if Sz12==0:
                    dd_state_indices.append(i)
                    print "dd_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2

    # Create Upp matrix for p-orbital multiplets
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out, dd_state_indices

def create_interaction_matrix_ALL_syms(VS,d_double,p_double,S_val, Sz_val, AorB_sym, A, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets including all symmetries
    
    Loop over all d_double states, find the corresponding sym channel; 
    the other loop over all d_double states, if it has same sym channel and S, Sz
    enter into the matrix element
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
        print "orbitals in sym ", sym, "= ", sym_orbs

        for i in d_double:
            # the state's orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            if state['type'] == 'one_hole_one_eh':
                se12 = state['e_spin']
                o1 = state['hole1_orb']
                o2 = state['hole2_orb']
                o12 = sorted([o1,o2])
                o12 = tuple(o12)

                # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
                S12  = S_val[i]
                Sz12 = Sz_val[i]

                # continue only if (o1,o2) is within desired sym
                if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
                    continue

                if (o1==o2=='dxz' or o1==o2=='dyz') and AorB_sym[i]!=AorB:
                    continue

                # get the corresponding index in sym for setting up matrix element
                idx1 = state_order[o12]
                
                # find state connecting to i
                for j in d_double:
                    if j<i:
                        continue
                        
                    state = VS.get_state(VS.lookup_tbl[j])
                    se34 = state['e_spin']
                    o3 = state['hole1_orb']
                    o4 = state['hole2_orb']
                    o34 = sorted([o3,o4])
                    o34 = tuple(o34)
                    S34  = S_val[j]
                    Sz34 = Sz_val[j]
                    
                    # see H_matrix_reduce_VS 
                    if se12!=se34:
                        continue

                    if (o3==o4=='dxz' or o3==o4=='dyz') and AorB_sym[j]!=AorB:
                        continue

                    # only same total spin S and Sz state have nonzero matrix element
                    if o34 in sym_orbs and S34==S12 and Sz34==Sz12:
                        idx2 = state_order[o34]

                        #print o12[0],o12[1],S12,Sz12," ",o34[0],o34[1],S34,Sz34," ", interaction_mat[idx1][idx2]
                        #print idx1, idx2

                        val = interaction_mat[idx1][idx2]
                        data.append(val); row.append(i); col.append(j)
                        if j!=i:
                            data.append(val); row.append(j); col.append(i)

                # get index for desired dd states
                # Note: for transformed basis of singlet/triplet
                # index can differ from that in original basis
                if 'dx2y2' in o12:
                    dd_state_indices.append(i)
                    #print "dd_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2

                # Special syms without b1 orbital:
                if (sym=='1B2' or sym=='3B2') and 'd3z2r2' in o12 and 'dxy' in o12:
                    dd_state_indices.append(i)
                    #print "dd_state_indices", i, ", state: S= ", S12, " Sz= ", Sz12, "orb= ", o1,o2
                 
    # Create Upp matrix for p-orbital multiplets
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out, dd_state_indices

def create_interaction_matrix_Norb3(VS,d_double,p_double, Udd, Upp):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets
    
    Cannot directly use the table in thesis or PRB paper since 
    those matrix elements are between singlets/triplets,
    need transform the basis in create_singlet_triplet_basis_change_matrix
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []
    pp_state_indices = []
    dp_state_indices = []

    for i in d_double:
        data.append(Udd); row.append(i); col.append(i)
        
        # get index for desired dd states
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']

        dd_state_indices.append(i)
        print "dd_state_indices", i, ", state: ", s1,o1,s2,o2
   
    for i in p_double:
        data.append(Upp); row.append(i); col.append(i)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out, dd_state_indices

def get_pp_state_indices(VS, S_val, Sz_val, AorB_sym):
    '''
    Get the list of index for desired pp and dp states for computing A(w)
    '''   
    dim = VS.dim
    pp_state_indices = []
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        if state['type'] == 'one_hole_one_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            o1 = state['hole1_orb']
            o2 = state['hole2_orb']
            x1, y1 = state['hole1_coord']
            x2, y2 = state['hole2_coord']

            if o1 in pam.O_orbs and o2 in pam.O_orbs and (x1,y1)==(0,1) and (x2,y2)==(1,0):
                # record singlet or triplet (only need Sz=1 state for calculating A(w))
                # note that this only works if basis_change_type = 'all_states' instead of 'd_double' in parameters.py
                if S_val[i]==0:
                    pp_state_indices.append(i)
                if S_val[i]==1 and Sz_val[i]==1:
                    pp_state_indices.append(i)
                #print "dp_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2
    
    return pp_state_indices

def get_dp_state_indices(VS, S_val, Sz_val, AorB_sym):
    '''
    Get the list of index for desired pp and dp states for computing A(w)
    '''   
    dim = VS.dim
    dp_state_indices = []
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        if state['type'] == 'one_hole_one_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            o1 = state['hole1_orb']
            o2 = state['hole2_orb']
            x1, y1 = state['hole1_coord']
            x2, y2 = state['hole2_coord']

            if o1 in pam.Cu_orbs and o2 in pam.O_orbs and (x1,y1)==(0,0) and (x2,y2)==(1,0):
                # record singlet or triplet (only need Sz=1 state for calculating A(w))
                # note that this only works if basis_change_type = 'all_states' instead of 'd_double' in parameters.py
                if S_val[i]==0:
                    dp_state_indices.append(i)
                    #print "dp_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2,S_val[i],Sz_val[i]
                if S_val[i]==1 and Sz_val[i]==1:
                    dp_state_indices.append(i)
                    #print "dp_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2,S_val[i],Sz_val[i]
    
    return dp_state_indices

def get_Cu_dx2y2_O_indices(VS, S_val, Sz_val, AorB_sym):
    '''
    Get the list of index for states with one hole on Cu and the other on neighboring O 
    with dx2-y2 symmetry (1/sqrt(4)) * (px1-py2-px3+py4)
    '''   
    dim = VS.dim
    Cu_dx2y2_O_indices = []
    
    for i in xrange(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        if state['type'] == 'one_hole_one_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            o1 = state['hole1_orb']
            o2 = state['hole2_orb']
            x1, y1 = state['hole1_coord']
            x2, y2 = state['hole2_coord']

            if o1!='dx2y2' and o2!='dx2y2':
                continue

            if o2 in pam.O_orbs and (x1,y1)==(0,0) and (x2,y2)==(1,0):
                # record singlet or triplet (only need Sz=1 state for calculating A(w))
                # note that this only works if basis_change_type = 'all_states' instead of 'd_double' in parameters.py
                if S_val[i]==0:
                    Cu_dx2y2_O_indices.append(i)
                if S_val[i]==1 and Sz_val[i]==0:
                    Cu_dx2y2_O_indices.append(i)
                #print "Cu_dx2y2_O_state_indices", i, ", state: ", s1,o1,x1,y1,s2,o2,x2,y2, 'S=',S_val[i],'Sz=',Sz_val[i]
            
    return Cu_dx2y2_O_indices

def check_dense_matrix_hermitian(matrix):
    '''
    Check if dense matrix is Hermitian. Returns True or False.
    '''
    dim = matrix.shape[0]
    out = True
    for row in range(0,dim):
        for col in range(0,dim):
            #if row==38 and col==85:
            #    print row, col, matrix[row,col], matrix[col,row]
            
            # sparse matrix has many zeros
            if abs(matrix[row,col])<1.e-10:
                continue
                
            if abs(matrix[row,col]-np.conjugate(matrix[col,row]))>1.e-10:
                print row, col, matrix[row,col], matrix[col,row]
                out = False
                break
    return out

def check_spin_group(row,col,data,VS):
    '''
    check if hoppings or interaction matrix occur within groups of (up,up), (dn,dn), and (up,dn) 
    since (up,up) state cannot hop to a (up,dn) or (dn,dn) state
    '''
    out = True
    dim = len(data)
    assert(len(row)==len(col)==len(data))
    
    for i in range(0,dim):
        irow = row[i]
        icol = col[i]
        
        rstate = VS.get_state(VS.lookup_tbl[irow])
        rs1 = rstate['hole1_spin']
        rs2 = rstate['hole2_spin']
        cstate = VS.get_state(VS.lookup_tbl[icol])
        cs1 = cstate['hole1_spin']
        cs2 = cstate['hole2_spin']
        
        rs = sorted([rs1,rs2])
        cs = sorted([cs1,cs2])
        
        if rs!=cs:
            ro1 = rstate['hole1_orb']
            ro2 = rstate['hole2_orb']
            rx1, ry1 = rstate['hole1_coord']
            rx2, ry2 = rstate['hole2_coord']
            
            co1 = cstate['hole1_orb']
            co2 = cstate['hole2_orb']
            cx1, cy1 = cstate['hole1_coord']
            cx2, cy2 = cstate['hole2_coord']
        
            print 'Error:'+str(rs)+' hops to '+str(cs)
            print 'Error occurs for state',irow,rs1,ro1,rx1,ry1,rs2,ro2,rx2,ry2, \
                  'hops to state',icol,cs1,co1,cx1,cy1,cs2,co2,cx2,cy2
            out = False
            break
    return out

def compare_matrices(m1,m2):
    '''
    Check if two matrices are the same. Returns True or False
    '''
    dim = m1.shape[0]
    if m2.shape[0] != dim:
        return False
    else:
        out = True
        for row in range(0,dim):
            for col in range(0,dim):
                if m1[row,col] != m2[row,col]:
                    out = False
                    break
        return out
        
if __name__ == '__main__':
    pass
