'''
1 el-hole pair must have opposite spins 
so only need to consider the case of electron spin up

Analysis of VS in terms of spin symmetry:
Because we allow 1 e-h pair at most, plus a vacuum state, VS consists of:
1. vac
2. eh pair (Nd up electron + paired dn hole)
2. eh pair (Nd dn electron + paired up hole)

Then Hamiltonian matrix looks
[0  A  A
At  M  0
At  0  M]

where 0 is the site energy of the single vacuum state; A is a block matrix with 1xn dimension and At is its transpose. So A depends on Ni-Nd hoppings.

M is nxn matrix depending on other hoppings and site energies. n is the number of states with one Nd electron (spin up/down) and one Ni or O hole (spin down/up). So now we want to get the GS by another matrix instead.

Mona:
I think in this case you'll get the right answer if you use
  [0         sqrt(2) A
sqrt(2) At       M]
for the "symmetric" sector, ie where you expect the contribution to the eigestates to have the same entries for the 
2nd and 3rd part of your vector. In other words, if the eigenvalue of the "3x3" matrix has entries (phi_1  phi_2 phi_3) for the three blocks, and if the eigenstates are such that phi_2=phi_3, then you can recast that "3x3" problem into a "2x2" one with the matrix written above, and the eigenstate is (phi_1  sqrt(2) phi_2) 

The sqrt(2) is necessary so you maintain the same normalization in both formulations, i.e. phi_1^2 + 2 phi_2^2=1.
And be careful that the eigenstate now returns sqrt(2)*phi_2, not phi_2. You'll need to adjust for it if you need phi_2 to  calculate expectation values.

Note that only tNiNd hopping matrix elements need to multiply sqrt(2) !!!

===========================================================================
Example:

Take A= 6.0 ep= 7.0  tpd= 1.5  tpp= 0.55  tNiNd= 0.5  tNdNd= 0.2  Upp= 0

Consider VS consisting of both up and dn spin of Nd electron:

start getting ground state
lowest eigenvalue of H = 
[-1.47733     2.89022777  2.89022777  2.89022777  2.89022777  2.89022777
  2.89022777  3.59487516  3.59487516  3.59487516]
eigenvalue =  -1.4773299981500763
Compute the weights in GS (lowest Aw peak)
no e-h state weight =  0.784729999378181
state  dn Nd_s -1 -1 0 d3z2r2 0 0 0 , weight =  0.007153403234568851
state  up Nd_s -1 -1 0 d3z2r2 0 0 0 , weight =  0.007153403234568853
state  dn Nd_s -1 -1 0 dxy 0 0 0 , weight =  0.01961750196345825
state  up Nd_s -1 -1 0 dxy 0 0 0 , weight =  0.019617501963458264
state  dn Nd_s -1 1 0 d3z2r2 0 0 0 , weight =  0.007153403234568861
state  up Nd_s -1 1 0 d3z2r2 0 0 0 , weight =  0.007153403234568856
state  dn Nd_s -1 1 0 dxy 0 0 0 , weight =  0.01961750196345827
state  up Nd_s -1 1 0 dxy 0 0 0 , weight =  0.019617501963458232
state  dn Nd_s 1 -1 0 d3z2r2 0 0 0 , weight =  0.007153403234568856
state  up Nd_s 1 -1 0 d3z2r2 0 0 0 , weight =  0.007153403234568832
state  dn Nd_s 1 -1 0 dxy 0 0 0 , weight =  0.019617501963458288
state  up Nd_s 1 -1 0 dxy 0 0 0 , weight =  0.01961750196345827
state  dn Nd_s 1 1 0 d3z2r2 0 0 0 , weight =  0.007153403234568844
state  up Nd_s 1 1 0 d3z2r2 0 0 0 , weight =  0.007153403234568861
state  dn Nd_s 1 1 0 dxy 0 0 0 , weight =  0.01961750196345827
state  up Nd_s 1 1 0 dxy 0 0 0 , weight =  0.01961750196345828

------------------------------------------------------------------------
Consider VS consisting of only up/dn spin of Nd electron:

start getting ground state
lowest eigenvalue of H = 
[-1.47733     2.89022777  2.89022777  2.89022777  2.89022777  3.59487516
  3.59487516  3.59487516  3.67948194  4.        ]
eigenvalue =  -1.4773299981500745
Compute the weights in GS (lowest Aw peak)
no e-h state weight =  0.78472999937818
state  up Nd_s -1 -1 0 d3z2r2 0 0 0 , weight =  0.014306806469137687
state  up Nd_s -1 -1 0 dxy 0 0 0 , weight =  0.03923500392691651
state  up Nd_s -1 1 0 d3z2r2 0 0 0 , weight =  0.014306806469137687
state  up Nd_s -1 1 0 dxy 0 0 0 , weight =  0.03923500392691652
state  up Nd_s 1 -1 0 d3z2r2 0 0 0 , weight =  0.014306806469137701
state  up Nd_s 1 -1 0 dxy 0 0 0 , weight =  0.03923500392691648
state  up Nd_s 1 1 0 d3z2r2 0 0 0 , weight =  0.014306806469137701
state  up Nd_s 1 1 0 dxy 0 0 0 , weight =  0.039235003926916534

'''