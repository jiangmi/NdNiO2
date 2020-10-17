'''
Simiar to the NiO2 two hole model, two holes can be up_up, up_dn, dn_up, dn_dn
only need to consider up_dn, dn_up (which can generate both singlet and triplet states) for reducing VS size

See vs.py:
----------
if pam.VS_only_up_dn==1:
    sss = sorted([se,s1,s2,s3])
    if sss!=['dn','dn','up','up']:
        continue
                                                                                                
=================================================================
Example No.1:
=================================================================
A= 6.0 ep= 7.0  tpd= 1.5  tpp= 0.55  tNiNd= 1.0  tNdNd= 0.3  Upp= 0

Consider VS consisting of all cases:
Mc= 2
VS.dim =  5129
No. of two_hole_no_eh states with count_upup, count_updn, count_dnup, count_dndn: 36 55 26 36

lowest eigenvalue of H = 
[-2.3284562  -1.62715115 -1.61083248 -1.58771344 -1.57103008 -1.1939659
 -1.1939659  -1.1939659  -1.1939659  -1.15067767]
eigenvalue =  -2.32845620078949
Compute the weights in GS (lowest Aw peak)
no e-h state  up px -1 0 0 dn dx2y2 0 0 0 , weight =  0.02927343621570099
no e-h state  dn px -1 0 0 up dx2y2 0 0 0 , weight =  0.027237789877909587
no e-h state  up py 0 -1 0 dn dx2y2 0 0 0 , weight =  0.02927343621571441
no e-h state  dn py 0 -1 0 up dx2y2 0 0 0 , weight =  0.027237789877919912
no e-h state  up dx2y2 0 0 0 dn dx2y2 0 0 0 , weight =  0.7494426881952387
no e-h state  up dx2y2 0 0 0 dn py 0 1 0 , weight =  0.027237935810621956
no e-h state  dn dx2y2 0 0 0 up py 0 1 0 , weight =  0.02915380687073733
no e-h state  up dx2y2 0 0 0 dn px 1 0 0 , weight =  0.027237935810628913
no e-h state  dn dx2y2 0 0 0 up px 1 0 0 , weight =  0.02915380687072955

=================================================================
Consider VS consisting of only up_dn and dn_up:

Mc= 2
VS.dim =  2273
No. of two_hole_no_eh states with count_upup, count_updn, count_dnup, count_dndn: 0 55 26 0

lowest eigenvalue of H = 
[-2.3284562  -1.61083248 -1.58771344 -1.1939659  -1.1939659  -1.15067767
 -1.15067767 -1.15067767 -1.10977223 -0.86538109]
eigenvalue =  -2.328456200789396
Compute the weights in GS (lowest Aw peak)
no e-h state  up px -1 0 0 dn dx2y2 0 0 0 , weight =  0.029273436215711896
no e-h state  dn px -1 0 0 up dx2y2 0 0 0 , weight =  0.027237789877910486
no e-h state  up py 0 -1 0 dn dx2y2 0 0 0 , weight =  0.029273436215710186
no e-h state  dn py 0 -1 0 up dx2y2 0 0 0 , weight =  0.027237789877915735
no e-h state  up dx2y2 0 0 0 dn dx2y2 0 0 0 , weight =  0.7494426881952416
no e-h state  up dx2y2 0 0 0 dn py 0 1 0 , weight =  0.027237935810631265
no e-h state  dn dx2y2 0 0 0 up py 0 1 0 , weight =  0.02915380687072632
no e-h state  up dx2y2 0 0 0 dn px 1 0 0 , weight =  0.027237935810623882
no e-h state  dn dx2y2 0 0 0 up px 1 0 0 , weight =  0.029153806870731644


=================================================================
Example No.2:
=================================================================
A= 3.0 ep= 7.0  tpd= 1.5  tpp= 0.55  tNiNd= 1.0  tNdNd= 0.3  Upp= 0

Consider VS consisting of all cases:

lowest eigenvalue of H = 
[-2.3284562  -1.62715115 -1.61083248 -1.58771344 -1.57103008 -1.1939659
 -1.1939659  -1.1939659  -1.1939659  -1.15067767]
eigenvalue =  -2.32845620078949
Compute the weights in GS (lowest Aw peak)
no e-h state  up px -1 0 0 dn dx2y2 0 0 0 , weight = 0.02927343621570099
no e-h state  dn px -1 0 0 up dx2y2 0 0 0 , weight =  0.027237789877909587
no e-h state  up py 0 -1 0 dn dx2y2 0 0 0 , weight =  0.02927343621571441
no e-h state  dn py 0 -1 0 up dx2y2 0 0 0 , weight =  0.027237789877919912
no e-h state  up dx2y2 0 0 0 dn dx2y2 0 0 0 , weight =  0.7494426881952387
no e-h state  up dx2y2 0 0 0 dn py 0 1 0 , weight =  0.027237935810621956
no e-h state  dn dx2y2 0 0 0 up py 0 1 0 , weight =  0.02915380687073733
no e-h state  up dx2y2 0 0 0 dn px 1 0 0 , weight =  0.027237935810628913
no e-h state  dn dx2y2 0 0 0 up px 1 0 0 , weight =  0.02915380687072955

=================================================================
Consider VS consisting of only up_dn and dn_up:

lowest eigenvalue of H = 
[-2.3284562  -1.61083248 -1.58771344 -1.1939659  -1.1939659  -1.15067767
 -1.15067767 -1.15067767 -1.10977223 -0.86538109]
eigenvalue =  -2.328456200789396
Compute the weights in GS (lowest Aw peak)
no e-h state  up px -1 0 0 dn dx2y2 0 0 0 , weight =  0.029273436215711896
no e-h state  dn px -1 0 0 up dx2y2 0 0 0 , weight =  0.027237789877910486
no e-h state  up py 0 -1 0 dn dx2y2 0 0 0 , weight =  0.029273436215710186
no e-h state  dn py 0 -1 0 up dx2y2 0 0 0 , weight =  0.027237789877915735
no e-h state  up dx2y2 0 0 0 dn dx2y2 0 0 0 , weight =  0.7494426881952416
no e-h state  up dx2y2 0 0 0 dn py 0 1 0 , weight =  0.027237935810631265
no e-h state  dn dx2y2 0 0 0 up py 0 1 0 , weight =  0.02915380687072632
no e-h state  up dx2y2 0 0 0 dn px 1 0 0 , weight =  0.027237935810623882
no e-h state  dn dx2y2 0 0 0 up px 1 0 0 , weight =  0.029153806870731644
'''