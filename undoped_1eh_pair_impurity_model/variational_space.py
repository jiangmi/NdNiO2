'''
Contains a class for the variational space for the cuprate layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between any two particles(holes) cannot > cutoff Mc

1 el-hole pair must have opposite spins 
so only need to consider the case of electron spin up

Analysis of VS in terms of spin symmetry:
Because we allow 1 e-h pair at most, plus a vacuum state, VS consists of:
1. vac
2. eh pair (Nd up electron + paired dn hole)
2. eh pair (Nd dn electron + paired up hole)

See vs_reduction_note.py for more details !
'''

import parameters as pam
import lattice as lat
import bisect
import numpy as np

def create_no_eh_state():
    '''
    Creates a dictionary representing the special vacuum state
    without any el-hole pairs
    '''
    state = {'type':'no_eh'}
    return state

def create_one_eh_state(s,eorb,xe,ye,ze,horb,xh,yh,zh):
    '''
    Creates a dictionary representing a state with 
    1 el-hole pair with opposite spins (only need to set e_spin)
    Here set general e_orb but in fact e orb has to be Nd
    holes can be on Ni or O

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    assert not (((xe,ye,ze))==(xh,yh,zh) and eorb==horb)
    assert(check_in_vs_condition(xe,ye,xh,yh))
    
    state = {'type'   :'one_eh',\
             'e_spin' : s,\
             'e_orb'  : eorb,\
             'e_coord': (xe,ye,ze),\
             'h_orb'  : horb,\
             'h_coord': (xh,yh,zh)}
    
    return state
    
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
            
            if state['type'] == 'no_eh':
                print i, 'no_eh state'
            if state['type'] == 'one_eh':    
                ts = state['e_spin']
                torb1 = state['e_orb']
                torb2 = state['h_orb']
                tx1, ty1, tz1 = state['e_coord']
                tx2, ty2, tz2 = state['h_coord']
                #if ts1=='up' and ts2=='up':
                #if torb1=='dx2y2' and torb2=='px':
                print i, ts,torb1,tx1,ty1,tz1,torb2,tx2,ty2,tz2
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Cu-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        lookup_tbl = []

        # vacuum undoped state (no e-h pairs)
        state = create_no_eh_state()

        if self.filter_func(state):
            uid = self.get_uid(state)
            lookup_tbl.append(uid)
        
        # one e-h pair
        # electron:
        for ux in range(-Mc,Mc+1):
            Bu = Mc - abs(ux)
            for uy in range(-Bu,Bu+1):
                for uz in [0]:
                    orb1s = lat.get_unit_cell_rep(ux,uy,uz)
                    
                    # el must be on Nd
                    if orb1s!=pam.Nd_orbs:
                        continue

                    # hole:
                    for vx in range(-Mc,Mc+1):
                        Bv = Mc - abs(vx)
                        for vy in range(-Bv,Bv+1):
                            for vz in [0]:
                                orb2s = lat.get_unit_cell_rep(vx,vy,vz)
                                
                                if orb2s==['NotOnSublattice'] or orb2s==pam.Nd_orbs:
                                    continue
                                if not check_in_vs_condition(ux,uy,vx,vy):
                                    continue

                                for orb1 in orb1s:
                                    for orb2 in orb2s:
                                        # only need to consider one spin case, see top
                                        for s in ['up']:#,'dn']:
                                            state = create_one_eh_state(s,orb1,ux,uy,uz,orb2,vx,vy,vz)

                                            if self.filter_func(state):
                                                uid = self.get_uid(state)
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
        if state['type'] == 'no_eh':
            return True
        elif state['type'] == 'one_eh':    
            x1, y1, z1 = state['e_coord']
            x2, y2, z2 = state['h_coord']

            if check_in_vs_condition(x1,y1,x2,y2):
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
        N2 = N*N

        if state['type'] == 'no_eh':
            uid = 0
            
        elif state['type'] == 'one_eh':
            off1 = 1 # can be any value larger than 0
            ss = state['e_spin']
            si = lat.spin_int[ss]
            orb1 = state['e_orb']
            orb2 = state['h_orb']
            o1 = lat.orb_int[orb1]
            o2 = lat.orb_int[orb2]
            x1, y1, z1 = state['e_coord']
            x2, y2, z2 = state['h_coord']

            # note the final +off1 to avoid mixing with no_eh state labeled as 0
            uid = si +4*z1 +8*z2 +16*o1 +16*N*o2 +16*N2*( (y1+s) + (x1+s)*B1 + (y2+s)*(B2+B1+1) + (x2+s)*(B3+B2+B1)*2) +off1

            # check if uid maps back to the original state, namely uid's uniqueness
            tstate = self.get_state(uid)
            ts = tstate['e_spin']
            torb1 = tstate['e_orb']
            torb2 = tstate['h_orb']
            tx1, ty1, tz1 = tstate['e_coord']
            tx2, ty2, tz2 = tstate['h_coord']
            assert((ss,orb1,x1,y1,z1,orb2,x2,y2,z2)==(ts,torb1,tx1,ty1,tz1,torb2,tx2,ty2,tz2))
            
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
        N2 = 16*N*N
        off1 = 1
        
        if uid < off1:
            state = create_no_eh_state()
        else:
            uid_ = uid - off1
            x2 = uid_/(N2*(B3+B2+B1)*2) - s
            uid_ = uid_ % (N2*(B3+B2+B1)*2)
            y2 = uid_/(N2*(B2+B1+1)) - s
            uid_ = uid_ % (N2*(B2+B1+1))
            x1 = uid_/(N2*B1) - s
            uid_ = uid_ % (N2*B1)
            y1 = uid_/N2 - s
            uid_ = uid_ % N2
            o2 = uid_/(16*N)
            uid_ = uid_ % (16*N)
            o1 = uid_/16 
            uid_ = uid_ % 16
            z2 = uid_/8 
            uid_ = uid_ % 8
            z1 = uid_/4 
            si = uid_ % 4

            orb2 = lat.int_orb[o2]
            orb1 = lat.int_orb[o1]
            ss = lat.int_spin[si]

            state = create_one_eh_state(ss,orb1,x1,y1,z1,orb2,x2,y2,z2)
            
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
