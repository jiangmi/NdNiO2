'''
Contains a class for the variational space for the NiO2 layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles(holes) cannot > cutoff Mc

George's email on Jul.6, 2021:
If indeed you cont have an electron hole pair as in the first case i.e. only 2 holes and not 3 holes plus an electron
then you cover all of space since you can have only triplets and singlets and S=0 or 1 and these states are all covered by
considering only the Sz=0 of the singlet and the triplet i.e. up down + or- down up.
However with 3 holes and 1 electron the three holes could be in a state S=3/2 or S=1/2 and the coupling with the electron 
could get you to S=2,1,0. However if there is an electron in Z then there must be a hole in dz2 so at least one of the 3 holes
must be dz2 and it must have the same spin as the electron in Z. 
As long as we neglect more than one electron hole pair and not more than 2 ligand holes then the above is correct. If we
consider also 3 ligand hole states say starting from d8x2z2, we can end up with d10Lx2Lz2Lz2Z but then the spin of one of the Lz2 must be the same as the spin of the electron in Z.
This should also limit your Hilbert space needed? I supppose that you are working in the space with S as a good quantum number i.e. two hole states are singlets or triplets and the only 3 hole states must involve a Z electron and a hole of z2 symmetry and with a spin in the same direction as the Z electron.

Previous analysis of VS in terms of spin symmetry:
Because we allow 1 e-h pair at most, plus a doped hole, VS consists of:
1. two up_up holes (same as NiO2 model considering two holes)
2. two up_dn holes (same as NiO2 model considering two holes)
3. two dn_up holes (same as NiO2 model considering two holes)
4. two dn_dn holes (same as NiO2 model considering two holes)

5.  two up_up holes + eh pair (Nd up electron + paired dn hole)
6.  two up_up holes + eh pair (Nd dn electron + paired up hole)
7.  two up_dn holes + eh pair (Nd up electron + paired dn hole)
8.  two up_dn holes + eh pair (Nd dn electron + paired up hole)
9.  two dn_up holes + eh pair (Nd up electron + paired dn hole)
10. two dn_up holes + eh pair (Nd dn electron + paired up hole)
11. two dn_dn holes + eh pair (Nd up electron + paired dn hole)
12. two dn_dn holes + eh pair (Nd dn electron + paired up hole)

To reduce VS size, see H_matrix_reducing_VS.pdf for more details !

The simple rule is starting from d8, d9L, d10L2 with up and dn spins 
(up up and dn dn can be neglected or precisely in other unconnected subspace of VS). 

Then in the presence of e-h pair, the only constraint for spin is that 
the four spins (1 el and 3 holes) should form a list of up up dn dn. 

Enforce e electron has up spin ?!
'''
import parameters as pam
import lattice as lat
import bisect
import numpy as np

def create_two_hole_no_eh_state(slabel):
    '''
    Creates a dictionary representing the two-hole state
    without el-hole pairs
    '''
    state = {'type': 'two_hole_no_eh',\
             'hole1_spin' : slabel[0],  \
             'hole1_orb'  : slabel[1],\
             'hole1_coord': (slabel[2],slabel[3],slabel[4]),\
             'hole2_spin' : slabel[5],  \
             'hole2_orb'  : slabel[6],\
             'hole2_coord': (slabel[7],slabel[8],slabel[9])}
    
    return state

def create_two_hole_one_eh_state(slabel):
    '''
    Creates a dictionary representing a state with 
    1 el-hole pair with opposite(!!!) spin in addition to another hole
    Here set general e_orb but in fact e orb has to be Nd
    holes can only be on Ni or O
    Therefore, (se=up,s1=up,s2=up) is not allowed because
    eh pair must have opposite spin
    
    Note: To reduce the VS size, without loss of generality, may assume
          Os electron has spin up and at least one hole spin down

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    se = slabel[0]; orbe = slabel[1]; xe = slabel[2]; ye = slabel[3]; ze = slabel[4];
    s1 = slabel[5]; orb1 = slabel[6]; x1 = slabel[7]; y1 = slabel[8]; z1 = slabel[9];
    s2 = slabel[10]; orb2 = slabel[11]; x2 = slabel[12]; y2 = slabel[13]; z2 = slabel[14];
    s3 = slabel[15]; orb3 = slabel[16]; x3 = slabel[17]; y3 = slabel[18]; z3 = slabel[19];
    
    assert not (((x3,y3,z3))==(x1,y1,z1) and s3==s1 and orb3==orb1)
    assert not (((x3,y3,z3))==(x2,y2,z2) and s3==s2 and orb3==orb2)
    assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
    #assert(check_in_vs_condition(x1,y1,x2,y2))
    #assert(check_in_vs_condition(xe,ye,x1,y1))
    #assert(check_in_vs_condition(xe,ye,x2,y2))
    
    if pam.eh_spin_def=='same':
        assert(s1==se or s2==se or s3==se)
    elif pam.eh_spin_def=='oppo':
        assert(s1!=se or s2!=se or s3!=se)
    
    state = {'type'    :'two_hole_one_eh',\
             'e_spin'  : se,\
             'e_orb'   : orbe,\
             'e_coord' : (xe,ye,ze),\
             'hole1_spin' : s1,\
             'hole1_orb'  : orb1,\
             'hole1_coord': (x1,y1,z1),\
             'hole2_spin' : s2,\
             'hole2_orb'  : orb2,\
             'hole2_coord': (x2,y2,z2),\
             'hole3_spin' : s3,\
             'hole3_orb'  : orb3,\
             'hole3_coord': (x3,y3,z3)}
    
    return state
    
def reorder_state(slabel):
    '''
    reorder the s, orb, coord's labeling a state to prepare for generating its canonical state
    Useful for three hole case especially !!!
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    # default
    state_label = slabel
    phase = 1.0
    
    if (x2,y2)<(x1,y1):
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        phase = -1.0

    # note that z1 can differ from z2 in the presence of apical pz orbital
    elif (x1,y1)==(x2,y2):           
        if s1==s2:
            o12 = list(sorted([orb1,orb2]))
            if o12[0]==orb2:
                state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
                phase = -1.0  
        elif s1=='dn' and s2=='up':
            state_label = ['up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1]
            phase = -1.0
            
    return state_label, phase
                
def make_state_canonical(state):
    '''
    1. There are a few cases to avoid having duplicate states.
    The sign change due to anticommuting creation operators should be 
    taken into account so that phase below has a negative sign
    =============================================================
    Case 1: 
    Note here is different from Mirko's version for only same spin !!
    Now whenever hole2 is on left of hole 1, switch them and
    order the hole coordinates in such a way that the coordinates 
    of the left creation operator are lexicographically
    smaller than those of the right.
    =============================================================
    Case 2: 
    If two holes locate on the same (x,y) sites (even if including apical pz with z=1)
    a) same spin state: 
      up, dxy,    (0,0), up, dx2-y2, (0,0)
    = up, dx2-y2, (0,0), up, dxy,    (0,0)
    need sort orbital order
    b) opposite spin state:
    only keep spin1 = up state
    
    Different from periodic lattice, the phase simply needs to be 1 or -1
    
    2. Besides, see emails with Mirko on Mar.1, 2018:
    Suppose Tpd|state_i> = |state_j> = phase*|canonical_state_j>, then 
    tpd = <state_j | Tpd | state_i> 
        = conj(phase)* <canonical_state_j | Tpp | state_i>
    
    so <canonical_state_j | Tpp | state_i> = tpd/conj(phase)
                                           = tpd*phase
    
    Because conj(phase) = 1/phase, *phase and /phase in setting tpd and tpp seem to give same results
    But need to change * or / in both tpd and tpp functions
    
    Similar for tpp
    '''
    
    # default:
    canonical_state = state
    phase = 1.0
    
    if state['type'] == 'two_hole_no_eh':  
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        
        tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
        slabel, ph = reorder_state(tlabel)
        canonical_state = create_two_hole_no_eh_state(slabel)
        phase *= ph

    elif state['type'] == 'two_hole_one_eh':        
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
        
        '''
        For three holes, the original candidate state is c_1*c_2*c_3|vac>
        To generate the canonical_state:
        1. reorder c_1*c_2 if needed to have a tmp12;
        2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
        3. reorder tmp12's 1st hole part and tmp23's 1st hole part
        '''
        tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
        tmp12,ph = reorder_state(tlabel)
        phase *= ph
        
        tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
        tmp23, ph = reorder_state(tlabel)
        phase *= ph
        
        tlabel = tmp12[0:5]+tmp23[0:5]
        tmp, ph = reorder_state(tlabel)
        phase *= ph
        
        slabel = [se,orbe,xe,ye,ze]+tmp+tmp23[5:10]
        canonical_state = create_two_hole_one_eh_state(slabel)
                
    return canonical_state, phase

def calc_manhattan_dist(x1,y1,x2,y2):
    '''
    Calculate the Manhattan distance (L1-norm) between two vectors
    (x1,y1) and (x2,y2).
    '''
    out = abs(x1-x2) + abs(y1-y2)
    return out

def check_in_vs_condition(x1,y1,x2,y2):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc:
        return False
    else:
        return True
    
def check_in_vs_condition1(x1,y1,x2,y2,x3,y3,x4,y4):
    '''
    Restrictions: the distance between one hole and Ni-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x3,y3,0,0) > pam.Mc or \
        calc_manhattan_dist(x4,y4,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc or \
        calc_manhattan_dist(x1,y1,x3,y3) > 2*pam.Mc or \
        calc_manhattan_dist(x1,y1,x4,y4) > 2*pam.Mc or \
        calc_manhattan_dist(x2,y2,x3,y3) > 2*pam.Mc or \
        calc_manhattan_dist(x2,y2,x4,y4) > 2*pam.Mc or \
        calc_manhattan_dist(x3,y3,x4,y4) > 2*pam.Mc:
        return False
    else:
        return True

def check_Pauli(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3):
    if (s1==s2 and orb1==orb2 and x1==x2 and y1==y2 and z1==z2) or \
        (s1==s3 and orb1==orb3 and x1==x3 and y1==y3 and z1==z3) or \
        (s3==s2 and orb3==orb2 and x3==x2 and y3==y2 and z3==z2):
        return False 
    else:
        return True
    
class VariationalSpace:
    '''
    Distance (L1-norm) between any two particles must not exceed a
    cutoff denoted by Mc. 

    Attributes
    ----------
    Mc: Cutoff for the hole-hole 
    lookup_tbl: sorted python list containing the unique identifiers 
        (uid) for all the states in the variational space. A uid is an
        integer which can be mapped to a state (see docsting of get_uid
        and get_state).
    dim: number of states in the variational space, i.e. length of
        lookup_tbl
    filter_func: a function that is passed to create additional 
        restrictions on the variational space. Default is None, 
        which means that no additional restrictions are implemented. 
        filter_func takes exactly one parameter which is a dictionary representing a state.

    Methods
    -------
    __init__
    create_lookup_table
    get_uid
    get_state
    get_index
    '''

    def __init__(self,Mc,filter_func=None):
        self.Mc = Mc
        if filter_func == None:
            self.filter_func = lambda x: True
        else:
            self.filter_func = filter_func
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print ("VS.dim = ", self.dim)
        #self.print_VS()

    def print_VS(self):
        for i in range(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])
            
            if state['type'] == 'two_hole_no_eh':
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                print ('no_eh state', i, s1,orb1,x1,y1,s2,orb2,x2,y2)
                
            elif state['type'] == 'two_hole_one_eh':  
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
                print ('one_eh state', i, se,orbe,xe,ye,ze,s1,orb1,x1,y1,s2,orb2,x2,y2,s3,orb3,x3,y3)
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Ni-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        lookup_tbl = []

        # two_hole no e-h pairs: hole can only be on Ni or O
        # hole1:
        for vx in range(-Mc,Mc+1):
            Bv = Mc - abs(vx)
            for vy in range(-Bv,Bv+1):
                orb1s = lat.get_unit_cell_rep(vx,vy,0)
                if orb1s==['NotOnSublattice'] or orb1s==pam.Ovacancy_orbs:
                    continue

                # hole2:
                for wx in range(-Mc,Mc+1):
                    Bw = Mc - abs(wx)
                    for wy in range(-Bw,Bw+1):
                        orb2s = lat.get_unit_cell_rep(wx,wy,0)
                        if orb2s==['NotOnSublattice'] or orb2s==pam.Ovacancy_orbs:
                            continue

                        if not check_in_vs_condition(vx,vy,wx,wy):
                            continue

                        for orb1 in orb1s:
                            for orb2 in orb2s:
                                for s1 in ['up','dn']:
                                    for s2 in ['up','dn']:   
                                        # see above G's email and also Mona's email on Jul.8,2021
                                        if pam.reduce_VS==1:
                                            if s1==s2:
                                                continue

                                        # consider Pauli principle
                                        if s1==s2 and orb1==orb2 and vx==wx and vy==wy:
                                            continue 

                                        slabel = [s1,orb1,vx,vy,0,s2,orb2,wx,wy,0]
                                        state = create_two_hole_no_eh_state(slabel)
                                        canonical_state,_ = make_state_canonical(state)

                                        if self.filter_func(canonical_state):
                                            uid = self.get_uid(canonical_state)
                                            lookup_tbl.append(uid)
           
        #print 'start generating states with one eh'
        # one e-h pair
        # electron:
        for ux in range(-Mc,Mc+1):
            Bu = Mc - abs(ux)
            for uy in range(-Bu,Bu+1):
                for uz in [-1,1]:
                    orbes = lat.get_unit_cell_rep(ux,uy,uz)
                    
                    # e must be on Nd
                    if orbes!=pam.Ovacancy_orbs:
                        continue

                    # hole1:
                    for vx in range(-Mc,Mc+1):
                        Bv = Mc - abs(vx)
                        for vy in range(-Bv,Bv+1):
                            orb1s = lat.get_unit_cell_rep(vx,vy,0)
                            if orb1s==['NotOnSublattice'] or orb1s==pam.Ovacancy_orbs:
                                continue

                            # hole2:
                            for wx in range(-Mc,Mc+1):
                                Bw = Mc - abs(wx)
                                for wy in range(-Bw,Bw+1):
                                    orb2s = lat.get_unit_cell_rep(wx,wy,0)
                                    if orb2s==['NotOnSublattice'] or orb2s==pam.Ovacancy_orbs:
                                        continue

                                    # hole3:
                                    for tx in range(-Mc,Mc+1):
                                        Bt = Mc - abs(tx)
                                        for ty in range(-Bt,Bt+1):
                                            orb3s = lat.get_unit_cell_rep(tx,ty,0)
                                            if orb3s==['NotOnSublattice'] or orb3s==pam.Ovacancy_orbs:
                                                continue

                                            if not check_in_vs_condition1(ux,uy,vx,vy,wx,wy,tx,ty):
                                                continue

                                            for orbe in orbes:
                                                for orb1 in orb1s:
                                                    for orb2 in orb2s:
                                                        for orb3 in orb3s:
                                                            for se in ['up','dn']:
                                                                for s1 in ['up','dn']:
                                                                    for s2 in ['up','dn']:   
                                                                        for s3 in ['up','dn']: 
                                                                            if pam.eh_spin_def=='same':
                                                                                # eh pair has same spin
                                                                                if s1!=se and s2!=se and s3!=se:
                                                                                    continue
                                                                            elif pam.eh_spin_def=='oppo':
                                                                                # eh pair has oppo spin
                                                                                if s1==se and s2==se and s3==se:
                                                                                    continue

                                                                            # neglect d7 state !!
                                                                            if orb1 in pam.Ni_orbs and \
                                                                                orb2 in pam.Ni_orbs and \
                                                                                orb3 in pam.Ni_orbs:
                                                                                continue

                                                                            # assume one hole is up (see no-eh case)
                                                                            if pam.reduce_VS==1:
                                                                                sss = sorted([se,s1,s2,s3])
                                                                                if pam.eh_spin_def=='same':
                                                                                    if sss!=['dn','dn','dn','up'] and \
                                                                                       sss!=['dn','up','up','up']:
                                                                                        continue
                                                                                elif pam.eh_spin_def=='oppo':
                                                                                    if sss!=['dn','dn','up','up']:
                                                                                        continue

                                                                            # consider Pauli principle
                                                                            if not check_Pauli(s1,orb1,vx,vy,0,\
                                                                                               s2,orb2,wx,wy,0,\
                                                                                               s3,orb3,tx,ty,0):
                                                                                continue 

                                                                            slabel = [se,orbe,ux,uy,uz,s1,orb1,vx,vy,0,s2,orb2,wx,wy,0,s3,orb3,tx,ty,0]

                                                                            state = create_two_hole_one_eh_state(slabel)
                                                                            canonical_state,_ = make_state_canonical(state)

                                                                            if self.filter_func(canonical_state):
                                                                                uid = self.get_uid(canonical_state)
                                                                                lookup_tbl.append(uid)

 
        lookup_tbl = list(set(lookup_tbl)) # remove duplicates
        lookup_tbl.sort()
        #print "\n lookup_tbl:\n", lookup_tbl
        return lookup_tbl
            
    def check_in_vs(self,state):
        '''
        Check if a given state is in VS

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.
        Mc: integer cutoff for the Manhattan distance.

        Returns
        -------
        Boolean: True or False
        '''
        assert(self.filter_func(state) in [True,False])
        if state['type'] == 'two_hole_no_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            
            if check_in_vs_condition(x1,y1,x2,y2):
                return True
            else:
                return False
            
        elif state['type'] == 'two_hole_one_eh': 
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

            if check_in_vs_condition1(xe,ye,x1,y1,x2,y2,x3,y3):
                return True
            else:
                return False

    def get_uid(self,state):
        '''
        Every state in the variational space is associated with a unique
        identifier (uid) which is an integer number.
        
        Rule for setting uid (example below but showing ideas):
        Assuming that i1, i2 can take the values -1 and +1. Make sure that uid is always larger or equal to 0. 
        So add the offset +1 as i1+1. Now the largest value that (i1+1) can take is (1+1)=2. 
        Therefore the coefficient in front of (i2+1) should be 3. This ensures that when (i2+1) is larger than 0, 
        it will be multiplied by 3 and the result will be larger than any possible value of (i1+1). 
        The coefficient in front of (o1+1) needs to be larger than the largest possible value of (i1+1) + 3*(i2+1). 
        This means that the coefficient in front of (o1+1) must be larger than (1+1) + 3*(1+1) = 8, 
        so you can choose 9 and you get (i1+1) + 3*(i2+1) + 9*(o1+1) and so on ....

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.

        Returns
        -------
        uid (integer) or None if the state is not in the variational space.
        '''
        # Need to check if the state is in the VS, because after hopping the state can be outside of VS
        if not self.check_in_vs(state):
            return None
        
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        B4 = B1*B3
        B5 = B1*B4
        B6 = B1*B5
        B7 = B1*B6
        N2 = N*N
        N3 = N2*N
        N4 = N3*N

        if state['type'] == 'two_hole_no_eh':
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
                
            i1 = lat.spin_int[s1]
            i2 = lat.spin_int[s2]
            o1 = lat.orb_int[orb1]
            o2 = lat.orb_int[orb2]
            
            # note the final +off1 to avoid mixing with no_eh state labeled as 0
            uid = i1 + 2*i2 +4*z1 +8*z2 +16*o1 +16*N*o2 \
                + 16*N2*( (y1+s) + (x1+s)*B1 + (y2+s)*B2 + (x2+s)*B3)
                
            #uid = i1 + 2*i2 + 4*z1 + 8*z2 + 16*o1 + 16*N*o2 \
            #    +16*N2*( (y1+s) + (x1+s)*B1 + (y2+s)*(B2+B1+1) + (x2+s)*(B3+B2+B1)*2 )

            # check if uid maps back to the original state, namely uid's uniqueness
            tstate = self.get_state(uid)
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            assert((s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)==(ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2))
            
        elif state['type'] == 'two_hole_one_eh':
            off1 = 16*N2*B4 # a bit bigger than needed
            
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
                
            ie = lat.spin_int[se]
            i1 = lat.spin_int[s1]
            i2 = lat.spin_int[s2]
            i3 = lat.spin_int[s3]
            oe = lat.orb_int[orbe]
            o1 = lat.orb_int[orb1]
            o2 = lat.orb_int[orb2]
            o3 = lat.orb_int[orb3]

            # note the final +off1 to avoid mixing with no_eh state labeled as 0
            uid = ie + 2*i1 + 4*i2 +8*i3 \
                + 16*(ze+1) +48*z1 +96*z2 +192*z3 \
                + 384*oe +384*N*o1 +384*N2*o2 +384*N3*o3 \
                + 384*N4*( (ye+s) + (xe+s)*B1 + (y1+s)*B2 + (x1+s)*B3 + (y2+s)*B4 + (x2+s)*B5 + (y3+s)*B6 + (x3+s)*B7) +off1

            # check if uid maps back to the original state, namely uid's uniqueness
            tstate = self.get_state(uid)
            tse = tstate['e_spin']
            torbe = tstate['e_orb']
            txe, tye, tze = tstate['e_coord']
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            ts3 = tstate['hole3_spin']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            torb3 = tstate['hole3_orb']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            tx3, ty3, tz3 = tstate['hole3_coord']
            assert((se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3)== \
                   (tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3))
            
        return uid

    def get_state(self,uid):
        '''
        Given a unique identifier, return the corresponding state. 
        ''' 
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        B4 = B1*B3
        B5 = B1*B4
        B6 = B1*B5
        B7 = B1*B6
        N2 = N*N
        N3 = N2*N
        N4 = N3*N
        off1 = 16*N2*B4
        #off1 = 16*N2*B1*(B3+B2+B1)*2
        
        if uid < off1:                
            uid_ = uid 
            #x2 = uid_/(16*N2*(B3+B2+B1)*2) - s
            #uid_ = uid_ % (16*N2*(B3+B2+B1)*2)
            #y2 = uid_/(16*N2*(B2+B1+1)) - s
            #uid_ = uid_ % (16*N2*(B2+B1+1))
            x2 = int(uid_/(16*N2*B3)) - s
            uid_ = uid_ % (16*N2*B3)
            y2 = int(uid_/(16*N2*B2)) - s
            uid_ = uid_ % (16*N2*B2)
            x1 = int(uid_/(16*N2*B1)) - s
            uid_ = uid_ % (16*N2*B1)
            y1 = int(uid_/(16*N2)) - s
            uid_ = uid_ % (16*N2)
            o2 = int(uid_/(16*N)) 
            uid_ = uid_ % (16*N)
            o1 = int(uid_/16) 
            uid_ = uid_ % 16
            z2 = int(uid_/8) 
            uid_ = uid_ % 8
            z1 = int(uid_/4) 
            uid_ = uid_ % 4
            i2 = int(uid_/2)
            i1 = uid_ % 2
            
            orb1 = lat.int_orb[o1]
            orb2 = lat.int_orb[o2]
            s1 = lat.int_spin[i1]
            s2 = lat.int_spin[i2]
            
            slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
            state = create_two_hole_no_eh_state(slabel)
        else:
            uid_ = uid - off1
            x3 = int(uid_/(384*N4*B7)) - s
            uid_ = uid_ % (384*N4*B7)
            y3 = int(uid_/(384*N4*B6)) - s
            uid_ = uid_ % (384*N4*B6)
            x2 = int(uid_/(384*N4*B5)) - s
            uid_ = uid_ % (384*N4*B5)
            y2 = int(uid_/(384*N4*B4)) - s
            uid_ = uid_ % (384*N4*B4)
            x1 = int(uid_/(384*N4*B3)) - s
            uid_ = uid_ % (384*N4*B3)
            y1 = int(uid_/(384*N4*B2)) - s
            uid_ = uid_ % (384*N4*B2)
            xe = int(uid_/(384*N4*B1)) - s
            uid_ = uid_ % (384*N4*B1)
            ye = int(uid_/(384*N4)) - s
            uid_ = uid_ % (384*N4)
            o3 = int(uid_/(384*N3)) 
            uid_ = uid_ % (384*N3)
            o2 = int(uid_/(384*N2)) 
            uid_ = uid_ % (384*N2)
            o1 = int(uid_/(384*N)) 
            uid_ = uid_ % (384*N)
            oe = int(uid_/384)
            uid_ = uid_ % 384
            z3 = int(uid_/192)
            uid_ = uid_ % 192
            z2 = int(uid_/96)
            uid_ = uid_ % 96
            z1 = int(uid_/48)
            uid_ = uid_ % 48
            ze = int(uid_/16)-1
            uid_ = uid_ % 16
            i3 = int(uid_/8)
            uid_ = uid_ % 8
            i2 = int(uid_/4)
            uid_ = uid_ % 4
            i1 = int(uid_/2) 
            ie = uid_ % 2

            orb3 = lat.int_orb[o3]
            orb2 = lat.int_orb[o2]
            orb1 = lat.int_orb[o1]
            orbe = lat.int_orb[oe]
            se = lat.int_spin[ie]
            s1 = lat.int_spin[i1]
            s2 = lat.int_spin[i2]
            s3 = lat.int_spin[i3]

            slabel = [se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
            state = create_two_hole_one_eh_state(slabel)
            
        return state

    def get_index(self,state):
        '''
        Return the index under which the state is stored in the lookup
        table.  These indices are consecutive and can be used to
        index, e.g. the Hamiltonian matrix

        Parameters
        ----------
        state: dictionary representing a state

        Returns
        -------
        index: integer such that lookup_tbl[index] = get_uid(state,Mc).
            If the state is not in the variational space None is returned.
        '''
        uid = self.get_uid(state)
        if uid == None:
            return None
        else:
            index = bisect.bisect_left(self.lookup_tbl,uid)
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None
