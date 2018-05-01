""" RHF-CCSD similarity transformed Hamiltonian """

__authors__ = "Ashutosh Kumar"
__credits__ = [
    "T. D. Crawford", "Daniel G. A. Smith", "Lori A. Burns", "Ashutosh Kumar"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import time
import numpy as np
import psi4
from utils import ndot


class HelperCCHbar(object):
    """
    This class builds pieces of the similarity transformed hamiltomian,
    Hbar = e^(-T)He^(T) = H + [H,T] + 1/2![[H,T],T] + 1/3![[[H,T],T],T] + 1/4![[[[H,T],T],T],T]
    which can be used quite conveniently to solve lambda equations, calculate  excitation energies 
    (EOM-CC sigma equations), CC response properties etc.. Spin orbitals expression of all Hbar 
    components are written in einstein notation in the doctrings of functions below. Ofcourse, 
    we are constructing <p * alpha | Hbar |q * alpha> (or beta) and 
    <p * alpha * q * beta | Hbar | r * alpha * s * beta> components here.

    References:  
    1. J. Gauss and J.F. Stanton, J. Chem. Phys., volume 103, pp. 3561-3577 (1995). 
    """

    def __init__(self, ccsd, memory=2):

        # Start of the cchbar class
        time_init = time.time()

        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }

        self.F = ccsd.F
        self.Dia = ccsd.Dia
        self.Dijab = ccsd.Dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2

        print('\nBuilding HBAR components ...')

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
            raise Exception('get_F: string %s must have 2 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_Loovv(self):
        tmp = self.get_MO('oovv').copy()
        self.Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return self.Loovv

    def build_Looov(self):
        tmp = self.get_MO('ooov').copy()
        self.Looov = 2.0 * tmp - tmp.swapaxes(0, 1)
        return self.Looov

    def build_Lvovv(self):
        tmp = self.get_MO('vovv').copy()
        self.Lvovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return self.Lvovv

    def build_tau(self):
        self.ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        self.ttau += tmp
        return self.ttau

    # F and W are the one and two body intermediates which appear in the CCSD
    # T1 and T2 equations. Please refer to helper_ccenergy file for more details.

    def build_Hov(self):
        """ <m|Hbar|e> = F_me = f_me + t_nf <mn||ef> """
        self.Hov = self.get_F('ov').copy()
        self.Hov += ndot('nf,mnef->me', self.t1, self.Loovv)
        return self.Hov

    def build_Hoo(self):
        """
            <m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me
                         + t_ne <mn||ie> + tau_inef <mn||ef>
        """
        self.Hoo = self.get_F('oo').copy()
        self.Hoo += ndot('ie,me->mi', self.t1, self.get_F('ov'))
        self.Hoo += ndot('ne,mnie->mi', self.t1, self.Looov)
        self.Hoo += ndot('inef,mnef->mi', self.build_tau(), self.Loovv)
        return self.Hoo

    def build_Hvv(self):
        """
            <a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me 
                         + t_mf <am||ef> - tau_mnfa <mn||fe>
        """
        self.Hvv = self.get_F('vv').copy()
        self.Hvv -= ndot('ma,me->ae', self.t1, self.get_F('ov'))
        self.Hvv += ndot('mf,amef->ae', self.t1, self.Lvovv)
        self.Hvv -= ndot('mnfa,mnfe->ae', self.build_tau(), self.Loovv)
        return self.Hvv

    def build_Hoooo(self):
        """ 
            <mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij> 
                           + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>
        """
        self.Hoooo = self.get_MO('oooo').copy()
        self.Hoooo += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        self.Hoooo += ndot('ie,mnej->mnij', self.t1, self.get_MO('oovo'))
        self.Hoooo += ndot('ijef,mnef->mnij', self.build_tau(),
                           self.get_MO('oovv'))
        return self.Hoooo

    def build_Hvvvv(self):
        """
            <ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef> 
                           - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>
        """
        self.Hvvvv = self.get_MO('vvvv').copy()
        self.Hvvvv -= ndot('mb,amef->abef', self.t1, self.get_MO('vovv'))
        self.Hvvvv -= ndot('ma,bmfe->abef', self.t1, self.get_MO('vovv'))
        self.Hvvvv += ndot('mnab,mnef->abef', self.build_tau(),
                           self.get_MO('oovv'))
        return self.Hvvvv

    def build_Hvovv(self):
        """ <am|Hbar|ef> = <am||ef> - t_na <nm||ef> """
        self.Hvovv = self.get_MO('vovv').copy()
        self.Hvovv -= ndot('na,nmef->amef', self.t1, self.get_MO('oovv'))
        return self.Hvovv

    def build_Hooov(self):
        """ <mn|Hbar|ie> = <mn||ie> + t_if <mn||fe> """
        self.Hooov = self.get_MO('ooov').copy()
        self.Hooov += ndot('if,mnfe->mnie', self.t1, self.get_MO('oovv'))
        return self.Hooov

    def build_Hovvo(self):
        """ 
            <mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef> 
                           - t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>
        """
        self.Hovvo = self.get_MO('ovvo').copy()
        self.Hovvo += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        self.Hovvo -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        self.Hovvo -= ndot('jnfb,nmfe->mbej', self.build_tau(),
                           self.get_MO('oovv'))
        self.Hovvo += ndot('jnbf,nmfe->mbej', self.t2, self.Loovv)
        return self.Hovvo

    def build_Hovov(self):
        """ 
            <mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je> 
                           - (t_jnfb + t_jf t_nb) <nm||ef>
        """
        self.Hovov = self.get_MO('ovov').copy()
        self.Hovov += ndot('jf,bmef->mbje', self.t1, self.get_MO('vovv'))
        self.Hovov -= ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
        self.Hovov -= ndot('jnfb,nmef->mbje', self.build_tau(),
                           self.get_MO('oovv'))
        return self.Hovov

    def build_Hvvvo(self):
        """
            <ab|Hbar|ei> = <ab||ei> - F_me t_miab + t_if Wabef + 0.5 * tau_mnab <mn||ei> 
                           - P(ab) t_miaf <mb||ef> - P(ab) t_ma {<mb||ei> - t_nibf <mn||ef>}
        """
        # <ab||ei>

        self.Hvvvo = self.get_MO('vvvo').copy()

        # - Fme t_miab

        self.Hvvvo -= ndot('me,miab->abei', self.get_F('ov'), self.t2)
        tmp = ndot('mnfe,mf->ne', self.Loovv, self.t1)
        self.Hvvvo -= ndot('niab,ne->abei', self.t2, tmp)

        # t_if Wabef

        self.Hvvvo += ndot('if,abef->abei', self.t1, self.get_MO('vvvv'))
        tmp = ndot('if,ma->imfa', self.t1, self.t1)
        self.Hvvvo -= ndot('imfa,mbef->abei', tmp, self.get_MO('ovvv'))
        self.Hvvvo -= ndot('imfb,amef->abei', tmp, self.get_MO('vovv'))
        tmp = ndot('mnef,if->mnei', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('mnab,mnei->abei', self.t2, tmp)
        tmp = ndot('if,ma->imfa', self.t1, self.t1)
        tmp1 = ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('imfa,mbef->abei', tmp, tmp1)

        # 0.5 * tau_mnab <mn||ei>

        self.Hvvvo += ndot('mnab,mnei->abei', self.build_tau(),
                           self.get_MO('oovo'))

        # - P(ab) t_miaf <mb||ef>

        self.Hvvvo -= ndot('imfa,mbef->abei', self.t2, self.get_MO('ovvv'))
        self.Hvvvo -= ndot('imfb,amef->abei', self.t2, self.get_MO('vovv'))
        self.Hvvvo += ndot('mifb,amef->abei', self.t2, self.Lvovv)

        # - P(ab) t_ma <mb||ei>

        self.Hvvvo -= ndot('mb,amei->abei', self.t1, self.get_MO('vovo'))
        self.Hvvvo -= ndot('ma,bmie->abei', self.t1, self.get_MO('voov'))

        # P(ab) t_ma * t_nibf <mn||ef>

        tmp = ndot('mnef,ma->anef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('infb,anef->abei', self.t2, tmp)
        tmp = ndot('mnef,ma->nafe', self.Loovv, self.t1)
        self.Hvvvo -= ndot('nifb,nafe->abei', self.t2, tmp)
        tmp = ndot('nmef,mb->nefb', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('niaf,nefb->abei', self.t2, tmp)
        return self.Hvvvo

    def build_Hovoo(self):
        """ 
            <mb|Hbar|ij> = <mb||ij> - Fme t_ijbe - t_nb Wmnij + 0.5 * tau_ijef <mb||ef> 
                           + P(ij) t_jnbe <mn||ie> + P(ij) t_ie {<mb||ej> - t_njbf <mn||ef>}
        """
        # <mb||ij>

        self.Hovoo = self.get_MO('ovoo').copy()

        # - Fme t_ijbe

        self.Hovoo += ndot('me,ijeb->mbij', self.get_F('ov'), self.t2)
        tmp = ndot('mnef,nf->me', self.Loovv, self.t1)
        self.Hovoo += ndot('me,ijeb->mbij', tmp, self.t2)

        # - t_nb Wmnij

        self.Hovoo -= ndot('nb,mnij->mbij', self.t1, self.get_MO('oooo'))
        tmp = ndot('ie,nb->ineb', self.t1, self.t1)
        self.Hovoo -= ndot('ineb,mnej->mbij', tmp, self.get_MO('oovo'))
        self.Hovoo -= ndot('jneb,mnie->mbij', tmp, self.get_MO('ooov'))
        tmp = ndot('nb,mnef->mefb', self.t1, self.get_MO('oovv'))
        self.Hovoo -= ndot('ijef,mefb->mbij', self.t2, tmp)
        tmp = ndot('ie,jf->ijef', self.t1, self.t1)
        tmp1 = ndot('nb,mnef->mbef', self.t1, self.get_MO('oovv'))
        self.Hovoo -= ndot('mbef,ijef->mbij', tmp1, tmp)

        # 0.5 * tau_ijef <mb||ef>

        self.Hovoo += ndot('ijef,mbef->mbij', self.build_tau(),
                           self.get_MO('ovvv'))

        # P(ij) t_jnbe <mn||ie>

        self.Hovoo -= ndot('ineb,mnej->mbij', self.t2, self.get_MO('oovo'))
        self.Hovoo -= ndot('jneb,mnie->mbij', self.t2, self.get_MO('ooov'))
        self.Hovoo += ndot('jnbe,mnie->mbij', self.t2, self.Looov)

        # P(ij) t_ie <mb||ej>

        self.Hovoo += ndot('je,mbie->mbij', self.t1, self.get_MO('ovov'))
        self.Hovoo += ndot('ie,mbej->mbij', self.t1, self.get_MO('ovvo'))

        # - P(ij) t_ie * t_njbf <mn||ef>

        tmp = ndot('ie,mnef->mnif', self.t1, self.get_MO('oovv'))
        self.Hovoo -= ndot('jnfb,mnif->mbij', self.t2, tmp)
        tmp = ndot('mnef,njfb->mejb', self.Loovv, self.t2)
        self.Hovoo += ndot('mejb,ie->mbij', tmp, self.t1)
        tmp = ndot('je,mnfe->mnfj', self.t1, self.get_MO('oovv'))
        self.Hovoo -= ndot('infb,mnfj->mbij', self.t2, tmp)
        return self.Hovoo


# End HelperCCHbar class
