"""
A simple python script to compute RHF-CCSD linear response function 
for calculating properties like dipole polarizabilities, optical
rotations etc. Equations were spin-adapted using the unitary group approach. 

References: 
1.  H. Koch and P. Jørgensen, J. Chem. Phys. volume 93, pp. 3333-3344 (1991).
2. Chapter 13, "Molecular Electronic-Structure Theory", Trygve Helgaker, 
   Poul Jørgensen and Jeppe Olsen, John Wiley & Sons Ltd.
"""

__authors__ = "Ashutosh Kumar"
__credits__ = ["Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import time
import numpy as np
import psi4
import sys
sys.path.append("../../../Coupled-Cluster/RHF")
from utils import ndot
from utils import helper_diis

class HelperCCPert(object):
    def __init__(self, name, pert, ccsd, hbar, cclambda, omega):

        # start of the ccpert class
        time_init = time.time()

        # Grabbing all the info from the wavefunctions passed
        self.pert = pert
        self.name = name
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc 
        self.mints = ccsd.mints
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        self.ttau  =  hbar.ttau
        self.Loovv =  hbar.Loovv
        self.Looov =  hbar.Looov
        self.Lvovv =  hbar.Lvovv
        self.Hov   =  hbar.Hov
        self.Hvv   =  hbar.Hvv
        self.Hoo   =  hbar.Hoo
        self.Hoooo =  hbar.Hoooo
        self.Hvvvv =  hbar.Hvvvv
        self.Hvovv =  hbar.Hvovv
        self.Hooov =  hbar.Hooov
        self.Hovvo =  hbar.Hovvo
        self.Hovov =  hbar.Hovov
        self.Hvvvo =  hbar.Hvvvo
        self.Hovoo =  hbar.Hovoo
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
        self.omega = omega

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

        # Build the denominators from diagonal elements of Hbar and omega
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()
        self.Dijab = self.Hoo.diagonal().reshape(-1, 1, 1, 1) + self.Hoo.diagonal().reshape(-1, 1, 1) - self.Hvv.diagonal().reshape(-1, 1) - self.Hvv.diagonal() 
        self.Dia += omega
        self.Dijab += omega
        
        # Guesses for X1 and X2 amplitudes
        self.x1 = self.build_Avo().swapaxes(0,1)/self.Dia
        self.pertbar_ijab = self.build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        self.x2 = self.pertbar_ijab.copy()
        self.x2 += self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3)
        self.x2 = self.x2/self.Dijab
       
        # Guesses for Y1 and Y2 amplitudes
        self.y1 =  2.0 * self.x1.copy() 
        self.y2 =  4.0 * self.x2.copy()    
        self.y2 -= 2.0 * self.x2.swapaxes(2,3)

        # Conventions used :    
        # occ orbitals  : i, j, k, l, m, n
        # virt orbitals : a, b, c, d, e, f
        # all oribitals : p, q, r, s, t, u, v

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 2 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]


    def get_pert(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_pert: string %s must have 2 elements.' % string)
        return self.pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_Aoo(self):
        Aoo = self.get_pert('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1, self.get_pert('ov'))
        return Aoo

    def build_Aov(self):
        Aov = self.get_pert('ov').copy()
        return Aov

    def build_Avo(self):
        Avo =  self.get_pert('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert('vv'), self.t1)
        Avo -= ndot('ma,mi->ai', self.t1, self.get_pert('oo'))
        Avo += ndot('miea,me->ai', self.t2, self.get_pert('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2, self.get_pert('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert('ov'))
        return Avo

    def build_Avv(self):
        Avv =  self.get_pert('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1, self.get_pert('ov'))
        return Avv

    def build_Aovoo(self):
        Aovoo = 0
        Aovoo += ndot('ijeb,me->mbij', self.t2, self.get_pert('ov'))
        return Aovoo

    def build_Avvvo(self):
        Avvvo = 0
        Avvvo -= ndot('miab,me->abei', self.t2, self.get_pert('ov'))
        return Avvvo

    def build_Avvoo(self):
        Avvoo = 0
        Avvoo += ndot('ijeb,ae->abij', self.t2, self.build_Avv())
        Avvoo -= ndot('mjab,mi->abij', self.t2, self.build_Aoo())
        return Avvoo

    def build_Goo(self, t2, y2):
        Goo = 0
        Goo += ndot('mjab,ijab->mi', t2, y2)
        return Goo

    def build_Gvv(self, y2, t2):
        Gvv = 0
        Gvv -= ndot('ijab,ijeb->ae', y2, t2)
        return Gvv

    def build_Zvv(self):
        Zvv = 0
        Zvv += ndot('amef,mf->ae', self.Hvovv, self.x1, prefactor=2.0)
        Zvv += ndot('amfe,mf->ae', self.Hvovv, self.x1, prefactor=-1.0)
        Zvv -= ndot('mnaf,mnef->ae', self.x2, self.Loovv)
        return Zvv

    def build_Zoo(self):
        Zoo = 0
        Zoo -= ndot('mnie,ne->mi', self.Hooov, self.x1, prefactor=2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, self.x1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.Loovv, self.x2)
        return Zoo


    def update_X(self):
        
        # X1 equations
        r_x1  = self.build_Avo().swapaxes(0,1).copy()
        r_x1 -= self.omega * self.x1.copy()
        r_x1 += ndot('ie,ae->ia', self.x1, self.Hvv)
        r_x1 -= ndot('mi,ma->ia', self.Hoo, self.x1)
        r_x1 += ndot('maei,me->ia', self.Hovvo, self.x1, prefactor=2.0)
        r_x1 += ndot('maie,me->ia', self.Hovov, self.x1, prefactor=-1.0)
        r_x1 += ndot('miea,me->ia', self.x2, self.Hov, prefactor=2.0)
        r_x1 += ndot('imea,me->ia', self.x2, self.Hov, prefactor=-1.0)
        r_x1 += ndot('imef,amef->ia', self.x2, self.Hvovv, prefactor=2.0)
        r_x1 += ndot('imef,amfe->ia', self.x2, self.Hvovv, prefactor=-1.0)
        r_x1 -= ndot('mnie,mnae->ia', self.Hooov, self.x2, prefactor=2.0)
        r_x1 -= ndot('nmie,mnae->ia', self.Hooov, self.x2, prefactor=-1.0)

        # X2 equations
        r_x2 = self.build_Avvoo().swapaxes(0,2).swapaxes(1,3).copy()
        r_x2 -= 0.5 * self.omega * self.x2
        r_x2 += ndot('ie,abej->ijab', self.x1, self.Hvvvo)
        r_x2 -= ndot('mbij,ma->ijab', self.Hovoo, self.x1)
        r_x2 += ndot('mi,mjab->ijab', self.build_Zoo(), self.t2)
        r_x2 += ndot('ijeb,ae->ijab', self.t2, self.build_Zvv())
        r_x2 += ndot('ijeb,ae->ijab', self.x2, self.Hvv)
        r_x2 -= ndot('mi,mjab->ijab', self.Hoo, self.x2)
        r_x2 += ndot('mnij,mnab->ijab', self.Hoooo, self.x2, prefactor=0.5)
        r_x2 += ndot('ijef,abef->ijab', self.x2, self.Hvvvv, prefactor=0.5)
        r_x2 -= ndot('imeb,maje->ijab', self.x2, self.Hovov)
        r_x2 -= ndot('imea,mbej->ijab', self.x2, self.Hovvo)
        r_x2 += ndot('miea,mbej->ijab', self.x2, self.Hovvo, prefactor=2.0)
        r_x2 += ndot('miea,mbje->ijab', self.x2, self.Hovov, prefactor=-1.0)

        old_x2 = self.x2.copy()

        # update X1 and X2
        self.x1 += r_x1/self.Dia
        tmp = r_x2/self.Dijab
        self.x2 += tmp + tmp.swapaxes(0,1).swapaxes(2,3)

        # Calcuate rms with the residual 
        rms = np.einsum('ia,ia->', r_x1/self.Dia, r_x1/self.Dia)
        rms += np.einsum('ijab,ijab->', old_x2 - self.x2, old_x2 - self.x2)
        return np.sqrt(rms)

    def inhomogenous_y2(self):

        # Inhomogenous terms appearing in Y2 equations
        r_y2  = ndot('ia,jb->ijab', self.l1, self.build_Aov(), prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.l1, self.build_Aov()) 
        r_y2 += ndot('ijeb,ea->ijab', self.l2, self.build_Avv())
        r_y2 -= ndot('im,mjab->ijab', self.build_Aoo(), self.l2)

        r_y2 -= np.einsum('mieb,me,ja->ijab', self.Loovv, self.x1, self.l1) 
        r_y2 -= np.einsum('ijae,me,mb->ijab', self.Loovv, self.x1, self.l1)
        r_y2 -= np.einsum('me,ie,jmba->ijab', self.x1, self.l1, self.Loovv)
        r_y2 += 2.0 * np.einsum('imae,me,jb->ijab', self.Loovv, self.x1, self.l1)

        r_y2 -=  np.einsum('me,ijeb,ma->ijab', self.x1, self.l2, self.Hov)                             
        r_y2 -=  np.einsum('ie,me,jmba->ijab', self.Hov, self.x1, self.l2)
        r_y2 -=  np.einsum('me,ijef,fmba->ijab', self.x1, self.l2, self.Hvovv) 
        r_y2 -=  np.einsum('fjea,me,imbf->ijab', self.Hvovv, self.x1, self.l2)
        r_y2 -=  np.einsum('fibe,me,jmfa->ijab', self.Hvovv, self.x1, self.l2)
        r_y2 +=  2.0 * np.einsum('ijfb,fmae,me->ijab', self.l2, self.Hvovv, self.x1)
        r_y2 -=  np.einsum('ijfb,fmea,me->ijab', self.l2, self.Hvovv, self.x1)
        r_y2 +=  2.0 * np.einsum('fiea,me,jmbf->ijab', self.Hvovv, self.x1, self.l2)
        r_y2 -=  np.einsum('fiae,me,jmbf->ijab', self.Hvovv, self.x1, self.l2)
        r_y2 +=  np.einsum('me,ineb,jmna->ijab', self.x1, self.l2, self.Hooov) 
        r_y2 +=  np.einsum('me,nieb,mjna->ijab', self.x1, self.l2, self.Hooov)
        r_y2 +=  np.einsum('jine,me,nmba->ijab', self.Hooov, self.x1, self.l2)
        r_y2 -=  2.0 * np.einsum('mina,me,njeb->ijab', self.Hooov, self.x1, self.l2)
        r_y2 -= -1.0 * np.einsum('imna,me,njeb->ijab', self.Hooov, self.x1, self.l2)
        r_y2 -=  2.0 * np.einsum('imne,me,jnba->ijab', self.Hooov, self.x1, self.l2)
        r_y2 -= -1.0 * np.einsum('mine,me,jnba->ijab', self.Hooov, self.x1, self.l2)

        r_y2 +=  0.5 * np.einsum('ijef,mnef,mnab->ijab', self.l2, self.x2, self.get_MO('oovv')) 
        r_y2 +=  np.einsum('mifb,mnef,jnae->ijab', self.l2, self.x2, self.get_MO('oovv'))
        r_y2 +=  np.einsum('imfb,mnef,njae->ijab', self.l2, self.x2, self.get_MO('oovv'))
        r_y2 +=  0.5 * np.einsum('ijfe,mnba,mnef->ijab', self.get_MO('oovv'), self.l2, self.x2)
        r_y2 -=  np.einsum('inae,mjfb,mnef->ijab', self.Loovv, self.l2, self.x2)
        r_y2 -=  ndot('in,jnba->ijab', self.build_Goo(self.Loovv, self.x2), self.l2) 
        r_y2 +=  ndot('ijfb,af->ijab', self.l2, self.build_Gvv(self.Loovv, self.x2))
        r_y2 +=  ndot('ijae,be->ijab', self.Loovv, self.build_Gvv(self.l2, self.x2))
        r_y2 -=  ndot('imab,jm->ijab', self.Loovv, self.build_Goo(self.l2, self.x2))
        r_y2 -=  np.einsum('nifb,mnef,mjea->ijab', self.l2, self.x2, self.Loovv)
        r_y2 +=  2.0 * np.einsum('imae,njfb,mnef->ijab', self.Loovv, self.l2, self.x2)

        return r_y2


    def inhomogenous_y1(self):
        
        # Inhomogenous terms appearing in Y1 equations
        r_y1 =  ndot('imae,me->ia', self.Loovv, self.x1, prefactor=2.0)
        r_y1 -= ndot('im,ma->ia', self.build_Aoo(), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Avv())
        r_y1 += ndot('imfe,feam->ia', self.l2, self.build_Avvvo())
        r_y1 -= ndot('ienm,mnea->ia', self.build_Aovoo(), self.l2, prefactor=0.5)
        r_y1 -= ndot('iemn,mnae->ia', self.build_Aovoo(), self.l2, prefactor=0.5)

        r_y1 -=  np.einsum('me,ie,ma->ia', self.x1, self.l1, self.Hov) 
        r_y1 -=  np.einsum('ie,me,ma->ia', self.Hov, self.x1, self.l1)
        r_y1 -=  2.0 * np.einsum('me,ne,mina->ia', self.x1,self.l1, self.Hooov)     
        r_y1 -= -1.0 * np.einsum('me,ne,imna->ia', self.x1,self.l1, self.Hooov)    
        r_y1 -=  2.0 * np.einsum('me,na,imne->ia', self.x1,self.l1, self.Hooov) 
        r_y1 -= -1.0 * np.einsum('me,na,mine->ia', self.x1,self.l1, self.Hooov)
        r_y1 +=  2.0 * np.einsum('me,if,fmae->ia', self.x1,self.l1, self.Hvovv) 
        r_y1 += -1.0 * np.einsum('me,if,fmea->ia', self.x1,self.l1, self.Hvovv)
        r_y1 +=  2.0 * np.einsum('me,mf,fiea->ia', self.x1,self.l1, self.Hvovv)     
        r_y1 += -1.0 * np.einsum('me,mf,fiae->ia', self.x1,self.l1, self.Hvovv)    

        r_y1 +=  2.0 * np.einsum('imae,mnef,nf->ia', self.Loovv, self.x2, self.l1)  
        r_y1 += -1.0 * np.einsum('imae,mnfe,nf->ia', self.Loovv, self.x2, self.l1)  
        r_y1 -= ndot('ni,na->ia', self.build_Goo(self.x2, self.Loovv), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Gvv(self.x2, self.Loovv))

        r_y1 -=  np.einsum('me,nief,mfna->ia', self.x1, self.l2, self.Hovov)
        r_y1 -=  np.einsum('ifne,me,nmaf->ia', self.Hovov, self.x1, self.l2)
        r_y1 -=  np.einsum('me,inef,mfan->ia', self.x1, self.l2, self.Hovvo)
        r_y1 -=  np.einsum('ifen,me,nmfa->ia', self.Hovvo, self.x1, self.l2)
        r_y1 +=  0.5 * np.einsum('fgae,me,imfg->ia', self.Hvvvv, self.x1, self.l2) 
        r_y1 +=  0.5 * np.einsum('fgea,me,imgf->ia', self.Hvvvv, self.x1, self.l2)
        r_y1 +=  0.5 * np.einsum('imno,me,onea->ia', self.Hoooo, self.x1, self.l2)
        r_y1 +=  0.5 * np.einsum('mino,me,noea->ia', self.Hoooo, self.x1, self.l2)

        ### 3-body terms
        tmp  =  ndot('nb,fb->nf', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('inaf,nf->ia', self.Loovv, tmp) 
        tmp  =  ndot('me,fa->mefa', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('mief,mefa->ia', self.Loovv, tmp)
        tmp  =  ndot('me,ni->meni', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('meni,mnea->ia', tmp, self.Loovv)
        tmp  =  ndot('jf,nj->fn', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('inaf,fn->ia', self.Loovv, tmp)
        ### 3-body terms over

        ### X2 * L2 terms 
        r_y1  -=  ndot('mi,ma->ia', self.build_Goo(self.x2, self.l2), self.Hov)  
        r_y1  +=  ndot('ie,ea->ia', self.Hov, self.build_Gvv(self.x2, self.l2)) 

        r_y1  -=  np.einsum('imfg,mnef,gnea->ia', self.l2, self.x2, self.Hvovv) 
        r_y1  -=  np.einsum('mifg,mnef,gnae->ia', self.l2, self.x2, self.Hvovv) 
        r_y1  -=  np.einsum('gief,mnga,mnef->ia', self.Hvovv, self.l2, self.x2) 
        r_y1  +=  2.0 * np.einsum('nifg,mnef,gmae->ia', self.l2, self.x2, self.Hvovv)
        r_y1  += -1.0 * np.einsum('nifg,mnef,gmea->ia', self.l2, self.x2, self.Hvovv)

        r_y1  -=  ndot('giea,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=2.0) 
        r_y1  -=  ndot('giae,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=-1.0)

        r_y1  +=  np.einsum('oief,mnef,mnoa->ia', self.l2, self.x2, self.Hooov)  
        r_y1  +=  np.einsum('inoe,mofa,mnef->ia', self.Hooov, self.l2, self.x2)
        r_y1  +=  np.einsum('miof,onea,nmfe->ia', self.Hooov, self.l2, self.x2)

        r_y1  -=  ndot('mioa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=2.0) 
        r_y1  -=  ndot('imoa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=-1.0) 

        r_y1  -=  2.0 * np.einsum('imoe,nofa,mnef->ia', self.Hooov, self.l2, self.x2)
        r_y1  -= -1.0 * np.einsum('mioe,nofa,mnef->ia', self.Hooov, self.l2, self.x2)

        return r_y1

    def update_Y(self):

        # Homogenous terms Y1 (identical in structure to lambda1 equations)
        r_y1 = self.im_y1.copy()
        r_y1  += 2.0 * self.build_Aov().copy()
        r_y1 += self.omega * self.y1
        r_y1 += ndot('ie,ea->ia', self.y1, self.Hvv)
        r_y1 -= ndot('im,ma->ia', self.Hoo, self.y1)
        r_y1 += ndot('ieam,me->ia', self.Hovvo, self.y1, prefactor=2.0)
        r_y1 += ndot('iema,me->ia', self.Hovov, self.y1, prefactor=-1.0)
        r_y1 += ndot('imef,efam->ia', self.y2, self.Hvvvo)
        r_y1 -= ndot('iemn,mnae->ia', self.Hovoo, self.y2)
        r_y1 -= ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=2.0)
        r_y1 -= ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=-1.0)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=-1.0)

        # Homogenous terms Y2 (identical in structure to lambda2 equations)
        r_y2 = self.im_y2.copy()
        r_y2 += 0.5 * self.omega * self.y2.copy()
        r_y2 += ndot('ia,jb->ijab', self.y1, self.Hov, prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.y1, self.Hov)
        r_y2 += ndot('ijeb,ea->ijab', self.y2, self.Hvv)
        r_y2 -= ndot('im,mjab->ijab', self.Hoo, self.y2)
        r_y2 += ndot('ijmn,mnab->ijab', self.Hoooo, self.y2, prefactor=0.5)
        r_y2 += ndot('ijef,efab->ijab', self.y2, self.Hvvvv, prefactor=0.5)
        r_y2 += ndot('ie,ejab->ijab', self.y1, self.Hvovv, prefactor=2.0)
        r_y2 += ndot('ie,ejba->ijab', self.y1, self.Hvovv, prefactor=-1.0)
        r_y2 -= ndot('mb,jima->ijab', self.y1, self.Hooov, prefactor=2.0)
        r_y2 -= ndot('mb,ijma->ijab', self.y1, self.Hooov, prefactor=-1.0)
        r_y2 += ndot('ieam,mjeb->ijab', self.Hovvo, self.y2, prefactor=2.0)
        r_y2 += ndot('iema,mjeb->ijab', self.Hovov, self.y2, prefactor=-1.0)
        r_y2 -= ndot('mibe,jema->ijab', self.y2, self.Hovov)
        r_y2 -= ndot('mieb,jeam->ijab', self.y2, self.Hovvo)
        r_y2 += ndot('ijeb,ae->ijab', self.Loovv, self.build_Gvv(self.y2, self.t2))
        r_y2 -= ndot('mi,mjab->ijab', self.build_Goo(self.t2, self.y2), self.Loovv)

        # update Y1 and Y2
        self.y1 += r_y1/self.Dia
        old_y2 = self.y2.copy()
        tmp = r_y2/self.Dijab    
        self.y2 += tmp + tmp.swapaxes(0,1).swapaxes(2,3) 

        # Calcuate rms from the residual 
        rms = np.einsum('ia,ia->', r_y1/self.Dia, r_y1/self.Dia)
        rms += np.einsum('ijab,ijab->', old_y2 - self.y2, old_y2 - self.y2)
        return np.sqrt(rms)

    def pseudoresponse(self, hand):
        polar1 = 0
        polar2 = 0
        if hand == 'right':
            z1 = self.x1 ; z2 = self.x2
        else:
            z1 = self.y1 ; z2 = self.y2

        # To match the pseudoresponse with PSI4
        polar1 += ndot('ia,ai->', z1, self.build_Avo(), prefactor=2.0)
        tmp = self.pertbar_ijab + self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3) 
        polar2 += ndot('ijab,ijab->', z2, tmp, prefactor=2.0)
        polar2 += ndot('ijba,ijab->', z2, tmp, prefactor=-1.0)

        return -2.0 * (polar1 + polar2)

    def solve(self, hand, r_conv=1.e-7, maxiter=100, max_diis=8, start_diis=1):

        ### Start of the solve routine 
        ccpert_tstart = time.time()
        
        # calculate the pseudoresponse from guess amplitudes
        pseudoresponse_old = self.pseudoresponse(hand)
        print("CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E " % (self.name, 0, pseudoresponse_old, -pseudoresponse_old))

        # Set up DIIS before iterations begin
        if hand == 'right':
            diis_object = helper_diis(self.x1, self.x2, max_diis)
        else:
            diis_object = helper_diis(self.y1, self.y2, max_diis)
            # calculate the inhomogenous terms of the left hand amplitudes equation before iterations begin
            self.im_y1 = self.inhomogenous_y1()
            self.im_y2 = self.inhomogenous_y2()

        # Iterate!
        for CCPERT_iter in range(1, maxiter + 1):

            # Residual build and update
            if hand == 'right':
                rms = self.update_X()
            else:
                rms = self.update_Y()

            # pseudoresponse with updated amplitudes
            pseudoresponse = self.pseudoresponse(hand)

            # Print CCPERT iteration information
            print('CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E   DIIS = %d' % (self.name, CCPERT_iter, pseudoresponse, (pseudoresponse - pseudoresponse_old), diis_object.diis_size))

            # Check convergence
            if (rms < r_conv):
                print('\nCCPERT_%s has converged in %.3f seconds!' % (self.name, time.time() - ccpert_tstart))
                return pseudoresponse

            # Update old pseudoresponse
            pseudoresponse_old = pseudoresponse

            #  Add the new error vector
            if hand == 'right':
                diis_object.add_error_vector(self.x1, self.x2)
            else:
                diis_object.add_error_vector(self.y1, self.y2)


            if CCPERT_iter >= start_diis:
                if hand == 'right':    
                    self.x1, self.x2 = diis_object.extrapolate(self.x1, self.x2)
                else:    
                    self.y1, self.y2 = diis_object.extrapolate(self.y1, self.y2)

# End HelperCCPert class

class HelperCCLinresp(object):

    def __init__(self, cclambda, ccpert_x, ccpert_y):

        # start of the cclinresp class 
        time_init = time.time()

        self.ccpert_x = ccpert_x
        self.ccpert_y = ccpert_y

        self.pert_x = ccpert_x.pert
        self.pert_y = ccpert_y.pert

        self.l1 = cclambda.l1
        self.l2 = cclambda.l2

        self.x1_x = ccpert_x.x1
        self.x2_x = ccpert_x.x2
        self.y1_x = ccpert_x.y1
        self.y2_x = ccpert_x.y2

        self.x1_y = ccpert_y.x1
        self.x2_y = ccpert_y.x2
        self.y1_y = ccpert_y.y1
        self.y2_y = ccpert_y.y2


    def linresp(self):
        self.polar1 = 0
        self.polar2 = 0
        self.polar1 += ndot("ai,ia->", self.ccpert_x.build_Avo(), self.y1_y)
        self.polar1 += ndot("abij,ijab->", self.ccpert_x.build_Avvoo(), self.y2_y, prefactor=0.5)
        self.polar1 += ndot("baji,ijab->", self.ccpert_x.build_Avvoo(), self.y2_y, prefactor=0.5)

        tmp = ndot('ia,jb->ijab', self.l1, self.ccpert_x.build_Aov())
        self.polar2 += ndot('ijab,ijab->', tmp, self.x2_y, prefactor=2.0)
        self.polar2 += ndot('ijab,ijba->', tmp, self.x2_y, prefactor=-1.0)

        tmp = ndot('ia,ic->ac', self.l1, self.x1_y)
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_x.build_Avv())
        tmp = ndot('ia,ka->ik', self.l1, self.x1_y)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_x.build_Aoo())

        tmp = ndot('ijbc,bcaj->ia', self.l2, self.ccpert_x.build_Avvvo())
        self.polar2 += ndot('ia,ia->', tmp, self.x1_y)

        tmp = ndot('ijab,kbij->ak', self.l2, self.ccpert_x.build_Aovoo())
        self.polar2 -= ndot('ak,ka->', tmp, self.x1_y, prefactor=0.5)

        tmp = ndot('ijab,kaji->bk', self.l2, self.ccpert_x.build_Aovoo())
        self.polar2 -= ndot('bk,kb->', tmp, self.x1_y, prefactor=0.5)

        tmp = ndot('ijab,kjab->ik', self.l2, self.x2_y)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_x.build_Aoo(), prefactor=0.5)

        tmp = ndot('ijab,kiba->jk', self.l2, self.x2_y,)
        self.polar2 -= ndot('jk,kj->', tmp, self.ccpert_x.build_Aoo(), prefactor=0.5)

        tmp = ndot('ijab,ijac->bc', self.l2, self.x2_y,)
        self.polar2 += ndot('bc,bc->', tmp, self.ccpert_x.build_Avv(), prefactor=0.5)

        tmp = ndot('ijab,ijcb->ac', self.l2, self.x2_y,)
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_x.build_Avv(), prefactor=0.5)

        self.polar2 += ndot("ia,ia->", self.ccpert_x.build_Aov(), self.x1_y, prefactor=2.0)

        return -1.0*(self.polar1 + self.polar2)

# End HelperCCLinresp class
