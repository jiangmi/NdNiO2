import math
import numpy as np
M_PI = math.pi

Mc = 2

# Note that Ni-d and O-p orbitals use hole language
# while Nd orbs use electron language
ed = {'d3z2r2': -0.45,\
      'dx2y2' : -1.77,\
      'dxy'   : -0.4,\
      'dxz'   : -0.35,\
      'dyz'   : -0.35}
ed = {'d3z2r2': 0,\
      'dx2y2' : 0,\
      'dxy'   : 0,\
      'dxz'   : 0,\
      'dyz'   : 0}
eNd = 4.0

eps = np.arange(7.0, 7.01, 1.0) #[3.5]#,3.5,4.5]
As = np.arange(6.0, 6.01, 1.0)
B = 0.15
C = 0.58
#As = np.arange(100, 100.1, 1.0)
#B = 0
#C = 0

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3x^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 8
if Norb==3 or Norb==8:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = np.linspace(1.5, 1.5, num=1, endpoint=True) #[0.25]
    #tpds = [2.001]
    tpps = [0.55]
    tNiNds = [0.5]
    tNdNds = [0.2]
elif Norb==10 or Norb==11 or Norb==12:    
    # pdp = sqrt(3)/4*pds so that tpd(b2)=tpd(b1)/2: see Eskes's thesis and 1990 paper
    # the values of pds and pdp between papers have factor of 2 difference
    # here use Eskes's thesis Page 4
    # also note that tpd ~ pds*sqrt(3)/2
    vals = np.linspace(1.5, 1.5, num=1, endpoint=True)
    pdss = np.asarray(vals)*2./np.sqrt(3)
    pdps = np.asarray(pdss)*np.sqrt(3)/4.
    #pdss = [1.5]
    #pdps = [0.7]
    tNiNds = [0.5]
    tNdNds = [0.2]
    #------------------------------------------------------------------------------
    # note that tpp ~ (pps+ppp)/2
    # because 3 or 7 orbital bandwidth is 8*tpp while 9 orbital has 4*(pps+ppp)
    pps = 0.9
    ppp = 0.2
    #pps = 0.00001
    #ppp = 0.00001

eta = 0.02
Lanczos_maxiter = 600

# restriction to reduce variational space
VS_only_up_Nd = 1

basis_change_type = 'all_states' # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_find_lowpeak = 0
if if_find_lowpeak==1:
    peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
    if_write_lowpeak_ep_tpd = 1
if_write_Aw = 0
if_savefig_Aw = 0

if_get_ground_state = 1
if if_get_ground_state==1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10
if_compute_Aw_dd_total = 0
if_compute_Aw_pp = 0
if_compute_Aw_dp = 0
if_compute_Aw_Cu_dx2y2_O = 0

if Norb==3:
    Ni_orbs = ['dx2y2']
else:
    Ni_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
    #Ni_orbs = ['dx2y2','d3z2r2']
    Nd_orbs = ['Nd_s']
    
if Norb==3 or Norb==8:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
    ap_orbs  = []
elif Norb==10:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
    ap_orbs  = []
elif Norb==11:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
    ap_orbs  = ['apz']
elif Norb==12:
    O1_orbs  = ['px1','py1','pz1']
    O2_orbs  = ['px2','py2','pz2']
    ap_orbs  = []
O_orbs = O1_orbs + O2_orbs + ap_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_orbs.sort()
Nd_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
ap_orbs.sort()
O_orbs.sort()
print "Ni_orbs = ", Ni_orbs
print "Nd_orbs = ", Nd_orbs
print "O1_orbs = ",  O1_orbs
print "O2_orbs = ",  O2_orbs
print "ap_orbs = ",  ap_orbs
orbs = Ni_orbs + Nd_orbs + O_orbs 
#assert(len(orbs)==Norb)
# ======================================================================
# Below for interaction matrix
Upps = [0]
if Norb==3:
    Udds = As+4*B+3*C
    #Udds = [100.]
else:
    interaction_sym = ['ALL']#,'3B1']#,'1A2','3A2','1B1','3B1','1E','3E']#,'1B2','3B2']
    print "turn on interactions for symmetries = ",interaction_sym
    
    if interaction_sym == ['ALL']:
        symmetries = ['1A1','1B1','3B1','1A2','3A2','1E','3E']
    else:
        symmetries = ['1A1']
    print "compute A(w) for symmetries = ",symmetries
