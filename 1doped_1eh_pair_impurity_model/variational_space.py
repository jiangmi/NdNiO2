'''
Contains a class for the variational space for the cuprate layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles(holes) cannot > cutoff Mc

Analysis of VS in terms of spin symmetry:
Because we allow 1 e-h pair at most, plus a doped hole, VS consists of:
1. up hole
2. dn hole
3. up + eh pair (Nd up electron + paired dn hole)
4. dn + eh pair (Nd up electron + paired dn hole)
5. up + eh pair (Nd dn electron + paired up hole)
6. dn + eh pair (Nd dn electron + paired up hole)

Hence, 1+3+5 couple together while 2+4+6 couple together
so that only need to consider 1+3+5 for reducing VS size

Then Hamiltonian matrix looks
[0  A  A
At  M  0
At  0  M]

where 0 is the site energy of the single vacuum state; A is a block matrix with 1xn dimension and At is its transpose. So A depends on Ni-Nd hoppings.

M is nxn matrix depending on other hoppings and site energies. n is the number of states with one Nd electron (spin up/down) and one Ni or O hole (spin down/up). So now we want to get the GS by another matrix instead.

Mona replied:
I think in this case you'll get the right answer is you use
  [0         sqrt(2) A
sqrt(2) At       M]
for the "symmetric" sector, ie where you expect the contribution to the eigestates to have the same entries for the 
2nd and 3rd part of your vector. In other words, if the eigenvalue of the "3x3" matrix has entries (phi_1  phi_2 phi_3) for the three blocks, and if the eigenstates are such that phi_2=phi_3, then you can recast that "3x3" problem into a "2x2" one with the matrix written above, and the eigenstate is (phi_1  sqrt(2) phi_2) 

The sqrt(2) is necessary so you maintain the same normalization in both formulations, i.e. phi_1^2 + 2 phi_2^2=1.
And be careful that the eigenstate now returns sqrt(2)*phi_2, not phi_2. You'll need to adjust for it if you need phi_2 to  calculate expectation values.
'''
import parameters as pam
import lattice as lat
import bisect
import numpy as np

def create_one_hole_no_eh_state(s,horb,xh,yh,zh):
    '''
    Creates a dictionary representing the special one-hole state
    without any el-hole pairs
    '''
    state = {'type'   :'one_hole_no_eh',\
             'spin'   : s,\
             'hole_orb'  : horb,\
             'hole_coord': (xh,yh,zh)}
    return state

def create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2):
    '''
    Creates a dictionary representing a state with 
    1 el-hole pair with opposite(!!!) spin in addition to another hole
    Here set general e_orb but in fact e orb has to be Nd
    holes can only be on Ni or O
    Therefore, (se=up,s1=up,s2=up) is not allowed because
    eh pair must have opposite spin
    
    Note: To reduce the VS size, without loss of generality, may assume
          Nd electron has spin up and at least one hole spin down

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    assert not (((xe,ye,ze))==(x1,y1,z1) and se==s1 and orbe==orb1)
    assert not (((xe,ye,ze))==(x2,y2,z2) and se==s2 and orbe==orb2)
    assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
    assert(check_in_vs_condition(x1,y1,x2,y2))
    assert(check_in_vs_condition(xe,ye,x1,y1))
    assert(check_in_vs_condition(xe,ye,x2,y2))
    assert(s1!=se or s2!=se)
    
    state = {'type'    :'one_hole_one_eh',\
             'e_spin'  : se,\
             'e_orb'   : orbe,\
             'e_coord' : (xe,ye,ze),\
             'hole1_spin' : s1,\
             'hole1_orb'  : orb1,\
             'hole1_coord': (x1,y1,z1),\
             'hole2_spin' : s2,\
             'hole2_orb'  : orb2,\
             'hole2_coord': (x2,y2,z2)}
    
    return state
    
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
    smaller than those of the right.Considering the constraints of two holes' spins,
    1. There are a few cases to avoid having duplicate states where 
    the holes are indistinguishable. 
    
    The sign change due to anticommuting creation operators should be 
    taken into account so that phase below has a negative sign
    =============================================================
    Case 1: 
    If two holes have same spin but different sites:
    when hole2 is on left of hole 1, switch them and
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
        
        if (x2,y2)<(x1,y1) and s1==s2:
            canonical_state = create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1)
            phase = -1.0

        # note that z1 can differ from z2 in the presence of apical pz orbital
        elif (x1,y1)==(x2,y2):           
            if s1==s2:
                o12 = list(sorted([orb1,orb2]))
                if o12[0]==orb2:
                    canonical_state = create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1)
                    phase = -1.0  
            elif s1=='dn' and s2=='up':
                canonical_state = create_one_hole_one_eh_state(se,orbe,xe,ye,ze,'up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1)
                phase = -1.0

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
        print "VS.dim = ", self.dim
        #self.print_VS()

    def print_VS(self):
        for i in xrange(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])
            
            if state['type'] == 'one_hole_no_eh':
                s = state['spin']
                o = state['hole_orb']
                x,y,z = state['hole_coord']
                print 'no_eh state', i, s,o,x,y
                
            elif state['type'] == 'one_hole_one_eh':  
                se = state['e_spin']
                orbe = state['e_orb']
                xe, ye, ze = state['e_coord']
                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                print 'one_eh state', i, se,orbe,xe,ye,s1,orb1,x1,y1,s2,orb2,x2,y2
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Cu-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list
        '''
        Mc = self.Mc
        lookup_tbl = []

        # one_hole no e-h pairs: hole can only be on Ni or O
        for ux in range(-Mc,Mc+1):
            Bu = Mc - abs(ux)
            for uy in range(-Bu,Bu+1):
                for uz in [0]:
                    orb1s = lat.get_unit_cell_rep(ux,uy,uz)
                    if orb1s==['NotOnSublattice'] or orb1s==pam.Nd_orbs:
                        continue
                        
                    for orb1 in orb1s:
                        # only need to consider up case, see top
                        for s1 in ['up','dn']:  
                            if check_in_vs_condition(ux,uy,0,0):
                                state = create_one_hole_no_eh_state(s1,orb1,ux,uy,uz)
                                canonical_state,_ = make_state_canonical(state)

                            if self.filter_func(canonical_state):
                                uid = self.get_uid(canonical_state)
                                lookup_tbl.append(uid)

        # one e-h pair
        # electron:
        for ux in range(-Mc,Mc+1):
            Bu = Mc - abs(ux)
            for uy in range(-Bu,Bu+1):
                for uz in [0]:
                    orbes = lat.get_unit_cell_rep(ux,uy,uz)
                    
                    # e must be on Nd
                    if orbes!=pam.Nd_orbs:
                        continue

                    # hole1:
                    for vx in range(-Mc,Mc+1):
                        Bv = Mc - abs(vx)
                        for vy in range(-Bv,Bv+1):
                            for vz in [0]:
                                orb1s = lat.get_unit_cell_rep(vx,vy,vz)
                                if orb1s==['NotOnSublattice'] or orb1s==pam.Nd_orbs:
                                    continue

                                # hole2:
                                for wx in range(-Mc,Mc+1):
                                    Bw = Mc - abs(wx)
                                    for wy in range(-Bw,Bw+1):
                                        for wz in [0]:
                                            orb2s = lat.get_unit_cell_rep(wx,wy,wz)
                                            if orb2s==['NotOnSublattice'] or orb2s==pam.Nd_orbs:
                                                continue

                                            if not (check_in_vs_condition(ux,uy,vx,vy) and \
                                                check_in_vs_condition(ux,uy,wx,wy) and \
                                                check_in_vs_condition(vx,vy,wx,wy)):
                                                continue
                                                        
                                            for orbe in orbes:
                                                for orb1 in orb1s:
                                                    for orb2 in orb2s:
                                                        for se in ['up','dn']:
                                                            for s1 in ['up','dn']:
                                                                for s2 in ['up','dn']:   
                                                                    # s2!=se because of eh pair
                                                                    if s2==se and s1==se:
                                                                        continue
    
                                                                    # try screen out same hole spin states
                                                                    if pam.VS_only_up_dn==1:
                                                                        if s1==s2:
                                                                            continue
                                                                    # try only keep Sz=1 triplet states
                                                                    if pam.VS_only_up_up==1:
                                                                        if not s1==s2=='up':
                                                                            continue

                                                                    # consider Pauli principle
                                                                    if s1==s2 and orb1==orb2 \
                                                                        and vx==wx and vy==wy and vz==wz:
                                                                        continue 
                                                                    if se==s1 and orbe==orb1 \
                                                                        and ux==vx and uy==vy and uz==vz:
                                                                        continue 
                                                                    if se==s2 and orbe==orb2 \
                                                                        and ux==wx and uy==wy and uz==wz:
                                                                        continue 

                                                                    state = create_one_hole_one_eh_state(se,orbe,ux,uy,uz,\
                                                                                                         s1,orb1,vx,vy,vz,s2,orb2,wx,wy,wz)
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
        if state['type'] == 'one_hole_no_eh':
            s = state['spin']
            x,y,z = state['hole_coord']
            if check_in_vs_condition(x,y,0,0):
                return True
            else:
                return False
            
        elif state['type'] == 'one_hole_one_eh':  
            xe, ye, ze = state['e_coord']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']

            if check_in_vs_condition(x1,y1,x2,y2) and \
                check_in_vs_condition(xe,ye,x1,y1) and \
                check_in_vs_condition(xe,ye,x2,y2):
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
        N2 = N*N
        N3 = N2*N

        if state['type'] == 'one_hole_no_eh':
            ss = state['spin']
            si = lat.spin_int[ss]
            orb = state['hole_orb']
            o1 = lat.orb_int[orb]
            x,y,z = state['hole_coord']
            uid = si +2*z +4*o1 +4*N*( (y+s) +B1*(x+s) )
            
            # check if uid maps back to the original state, namely uid's uniqueness
            tstate = self.get_state(uid)
            tss = tstate['spin']
            torb = tstate['hole_orb']
            tx,ty,tz = tstate['hole_coord']
            assert((ss,orb,x,y,z)==(tss,torb,tx,ty,tz))
            
        elif state['type'] == 'one_hole_one_eh':
            off1 = 4*N*B2 # a bit bigger than needed
            
            se = state['e_spin']
            orbe = state['e_orb']
            xe, ye, ze = state['e_coord']
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
                
            ie = lat.spin_int[se]
            i1 = lat.spin_int[s1]
            i2 = lat.spin_int[s2]
            oe = lat.orb_int[orbe]
            o1 = lat.orb_int[orb1]
            o2 = lat.orb_int[orb2]

            # note the final +off1 to avoid mixing with no_eh state labeled as 0
            uid = ie + 2*i1 + 4*i2 \
                + 8*ze +16*z1 +32*z2 \
                + 64*oe +64*N*o1 +64*N2*o2 \
                + 64*N3*( (ye+s) + (xe+s)*B1 + (y1+s)*B2 + (x1+s)*B3 + (y2+s)*B4 + (x2+s)*B5) +off1

            # check if uid maps back to the original state, namely uid's uniqueness
            tstate = self.get_state(uid)
            tse = tstate['e_spin']
            torbe = tstate['e_orb']
            txe, tye, tze = tstate['e_coord']
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            assert((se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)== \
                   (tse,torbe,txe,tye,tze,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2))
            
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
        N2 = N*N
        N3 = N2*N
        off1 = 4*N*B2 
        
        if uid < off1:
            uid_ = uid 
            x = uid_/(4*N*B1) - s
            uid_ = uid_ % (4*N*B1)
            y = uid_/(4*N) - s
            uid_ = uid_ % (4*N)
            o1 = uid_/4 
            uid_ = uid_ % 4
            z = uid_/2 
            si = uid_ % 2
            
            orb1 = lat.int_orb[o1]
            ss = lat.int_spin[si]
            state = create_one_hole_no_eh_state(ss,orb1,x,y,z)
        else:
            uid_ = uid - off1
            x2 = uid_/(64*N3*B5) - s
            uid_ = uid_ % (64*N3*B5)
            y2 = uid_/(64*N3*B4) - s
            uid_ = uid_ % (64*N3*B4)
            x1 = uid_/(64*N3*B3) - s
            uid_ = uid_ % (64*N3*B3)
            y1 = uid_/(64*N3*B2) - s
            uid_ = uid_ % (64*N3*B2)
            xe = uid_/(64*N3*B1) - s
            uid_ = uid_ % (64*N3*B1)
            ye = uid_/(64*N3) - s
            uid_ = uid_ % (64*N3)
            o2 = uid_/(64*N2) 
            uid_ = uid_ % (64*N2)
            o1 = uid_/(64*N) 
            uid_ = uid_ % (64*N)
            oe = uid_/64
            uid_ = uid_ % 64
            z2 = uid_/32
            uid_ = uid_ % 32
            z1 = uid_/16
            uid_ = uid_ % 16
            ze = uid_/8
            uid_ = uid_ % 8
            i2 = uid_/4
            uid_ = uid_ % 4
            i1 = uid_/2 
            ie = uid_ % 2

            orb2 = lat.int_orb[o2]
            orb1 = lat.int_orb[o1]
            orbe = lat.int_orb[oe]
            se = lat.int_spin[ie]
            s1 = lat.int_spin[i1]
            s2 = lat.int_spin[i2]

            state = create_one_hole_one_eh_state(se,orbe,xe,ye,ze,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)
            
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
