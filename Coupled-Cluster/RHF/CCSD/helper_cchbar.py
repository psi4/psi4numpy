"""
A simple python script to build pieces of the similarity
transformed hamiltomian required to build the RHF-CCSD Jacobian,
used in solving lambda equations, calculations of excitation 
energies (EOM-CCSD), CCSD response properties etc..
Terms were spin-adapted using the unitary group approach. 

References: 
1. Chapter 13, "Molecular Electronic-Structure Theory", Trygve Helgaker, 
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
from utils import ndot

class HelperCCHbar(object):

    def __init__(self, ccsd, memory=2):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}


        self.F = ccsd.F
        self.Dia = ccsd.Dia
        self.Dijab = ccsd.Dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2

        print('\nBuilding HBAR ...')

        self.build_Loovv() 
        self.build_Looov() 
        self.build_Lvovv() 

        self.build_Hov()
        self.build_Hoo()
        self.build_Hvv()
        self.build_Hoooo()
        self.build_Hvvvv()
        self.build_Hvovv()
        self.build_Hooov()
        self.build_Hovvo()
        self.build_Hovov()
        self.build_Hvvvo()
        self.build_Hovoo()

        print('\n..HBAR Build completed !!')

    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                     self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_Loovv(self):
        tmp = self.get_MO('oovv').copy()
        self.Loovv = 2.0 * tmp - tmp.swapaxes(2,3)
        return self.Loovv

    def build_Looov(self):
        tmp = self.get_MO('ooov').copy()
        self.Looov = 2.0 * tmp - tmp.swapaxes(0,1)
        return self.Looov

    def build_Lvovv(self):
        tmp = self.get_MO('vovv').copy()
        self.Lvovv = 2.0 * tmp - tmp.swapaxes(2,3)
        return self.Lvovv

    def build_tau(self):
        self.ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        self.ttau += tmp
        return self.ttau

    def build_Hov(self):
        self.Hov = self.get_F('ov').copy()
        self.Hov += ndot('nf,mnef->me', self.t1, self.Loovv)
        return self.Hov

    def build_Hoo(self):
        self.Hoo = self.get_F('oo').copy()
        self.Hoo += ndot('ie,me->mi', self.t1, self.get_F('ov'))
        self.Hoo += ndot('ne,mnie->mi', self.t1, self.Looov)
        self.Hoo += ndot('mnef,inef->mi', self.Loovv, self.build_tau())
        return self.Hoo

    def build_Hvv(self):
        self.Hvv = self.get_F('vv').copy()
        self.Hvv -= ndot('ma,me->ae', self.t1, self.get_F('ov'))
        self.Hvv += ndot('amef,mf->ae', self.Lvovv, self.t1)
        self.Hvv -= ndot('mnfa,mnfe->ae', self.build_tau(), self.Loovv)
        return self.Hvv

    def build_Hoooo(self):
        self.Hoooo = self.get_MO('oooo').copy()
        self.Hoooo += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'), prefactor=2.0)
        self.Hoooo += ndot('mnef,ijef->mnij', self.get_MO('oovv'), self.build_tau())
        return self.Hoooo

    def build_Hvvvv(self):
        self.Hvvvv = self.get_MO('vvvv').copy()
        self.Hvvvv -= ndot('mb,amef->abef', self.t1, self.get_MO('vovv'), prefactor=2.0)
        self.Hvvvv += ndot('mnab,mnef->abef', self.build_tau(), self.get_MO('oovv'))
        return self.Hvvvv

    def build_Hvovv(self):
        self.Hvovv = self.get_MO('vovv').copy()
        self.Hvovv -= ndot('na,nmef->amef', self.t1, self.get_MO('oovv'))
        return self.Hvovv

    def build_Hooov(self):
        self.Hooov = self.get_MO('ooov').copy()
        self.Hooov += ndot('if,nmef->mnie', self.t1, self.get_MO('oovv'))
        return self.Hooov

    def build_Hovvo(self):
        self.Hovvo = self.get_MO('ovvo').copy()
        self.Hovvo += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        self.Hovvo -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        self.Hovvo -= ndot('njbf,mnef->mbej', self.build_tau(), self.get_MO('oovv'))
        self.Hovvo += ndot('njfb,mnef->mbej', self.t2, self.Loovv)
        return self.Hovvo

    def build_Hovov(self):
        self.Hovov = self.get_MO('ovov').copy()
        self.Hovov += ndot('jf,bmef->mbje', self.t1, self.get_MO('vovv'))
        self.Hovov -= ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
        self.Hovov -= ndot('jnfb,nmef->mbje', self.build_tau(), self.get_MO('oovv'))
        return self.Hovov

    def build_Hvvvo(self):
        self.Hvvvo =  self.get_MO('vvvo').copy()
        self.Hvvvo += ndot('if,abef->abei', self.t1, self.get_MO('vvvv'))
        self.Hvvvo -= ndot('mb,amei->abei', self.t1, self.get_MO('vovo'))
        self.Hvvvo -= ndot('ma,bmie->abei', self.t1, self.get_MO('voov'))
        self.Hvvvo -= ndot('imfa,mbef->abei', self.build_tau(), self.get_MO('ovvv'))
        self.Hvvvo -= ndot('imfb,amef->abei', self.build_tau(), self.get_MO('vovv'))
        self.Hvvvo += ndot('mnab,mnei->abei', self.build_tau(), self.get_MO('oovo'))
        self.Hvvvo -= ndot('me,miab->abei', self.get_F('ov'), self.t2)
        self.Hvvvo += ndot('mifb,amef->abei', self.t2, self.Lvovv)
        tmp =    ndot('mnef,if->mnei', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('mnab,mnei->abei', self.t2, tmp)
        tmp =    ndot('mnef,ma->anef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('infb,anef->abei', self.t2, tmp)
        tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('miaf,mefb->abei', self.t2, tmp)
        tmp =    ndot('mnfe,mf->ne', self.Loovv, self.t1)
        self.Hvvvo -= ndot('niab,ne->abei', self.t2, tmp)
        tmp =    ndot('mnfe,na->mafe', self.Loovv, self.t1)
        self.Hvvvo -= ndot('mifb,mafe->abei', self.t2, tmp)
        tmp1 =   ndot('if,ma->imfa', self.t1, self.t1)
        tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('imfa,mbef->abei', tmp1, tmp2)
        return self.Hvvvo

    def build_Hovoo(self):
        self.Hovoo =  self.get_MO('ovoo').copy()
        self.Hovoo += ndot('mbie,je->mbij', self.get_MO('ovov'), self.t1)
        self.Hovoo += ndot('ie,bmje->mbij', self.t1, self.get_MO('voov'))
        self.Hovoo -= ndot('nb,mnij->mbij', self.t1, self.get_MO('oooo'))
        self.Hovoo -= ndot('ineb,nmje->mbij', self.build_tau(), self.get_MO('ooov'))
        self.Hovoo -= ndot('jneb,mnie->mbij', self.build_tau(), self.get_MO('ooov'))
        self.Hovoo += ndot('ijef,mbef->mbij', self.build_tau(), self.get_MO('ovvv'))
        self.Hovoo += ndot('me,ijeb->mbij', self.get_F('ov'), self.t2)
        self.Hovoo += ndot('njeb,mnie->mbij', self.t2, self.Looov)
        tmp =    ndot('mnef,jf->mnej', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('ineb,mnej->mbij', self.t2, tmp)
        tmp =    ndot('mnef,ie->mnif', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('jnfb,mnif->mbij', self.t2, tmp)
        tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('ijef,mefb->mbij', self.t2, tmp)
        tmp =    ndot('mnef,njfb->mejb', self.Loovv, self.t2)
        self.Hovoo += ndot('mejb,ie->mbij', tmp, self.t1)
        tmp =    ndot('mnef,nf->me', self.Loovv, self.t1)
        self.Hovoo += ndot('me,ijeb->mbij', tmp, self.t2)
        tmp1 =   ndot('ie,jf->ijef', self.t1, self.t1)
        tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('mbef,ijef->mbij', tmp2, tmp1)
        return self.Hovoo

# End HelperCCHbar class
