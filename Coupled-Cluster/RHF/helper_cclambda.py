# -*- coding: utf-8 -*-
"""
A simple python script to calculate RHF-CCSD lambda amplitudes
using pieces of the similarity transformed hamiltonian, Hbar.
Equations were spin-adapted using the unitary group approach. 

References: 
1. J. Gauss and J.F. Stanton, J. Chem. Phys., volume 103, pp. 3561-3577 (1995). 
2. Chapter 13, "Molecular Electronic-Structure Theory", Trygve Helgaker, 
   Poul Jørgensen and Jeppe Olsen, John Wiley & Sons Ltd.
3. Dr. Crawford's notes on commutators relationships of H with singles excitation 
   operators. (pdf in the current folder)
    
"""

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
from utils import helper_diis


class HelperCCLambda(object):
    def __init__(self, ccsd, hbar):

        # start of the cclambda class
        time_init = time.time()

        # Grabbing all the info from the wavefunctions passed
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc
        self.F = ccsd.F
        self.Dia = ccsd.Dia
        self.Dijab = ccsd.Dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        self.ttau = hbar.ttau
        self.Loovv = hbar.Loovv
        self.Looov = hbar.Looov
        self.Lvovv = hbar.Lvovv
        self.Hov = hbar.Hov
        self.Hvv = hbar.Hvv
        self.Hoo = hbar.Hoo
        self.Hoooo = hbar.Hoooo
        self.Hvvvv = hbar.Hvvvv
        self.Hvovv = hbar.Hvovv
        self.Hooov = hbar.Hooov
        self.Hovvo = hbar.Hovvo
        self.Hovov = hbar.Hovov
        self.Hvvvo = hbar.Hvvvo
        self.Hovoo = hbar.Hovoo

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }

        # Guesses for L1 and L2 amplitudes
        self.l1 = 2.0 * self.t1.copy()
        self.l2 = 4.0 * self.t2.copy()
        self.l2 -= 2.0 * self.t2.swapaxes(2, 3)

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

    def build_Goo(self):
        self.Goo = 0
        self.Goo += ndot('mjab,ijab->mi', self.t2, self.l2)
        return self.Goo

    def build_Gvv(self):
        self.Gvv = 0
        self.Gvv -= ndot('ijab,ijeb->ae', self.l2, self.t2)
        return self.Gvv

    def update(self):
        """
            l1 and l2  equations can be obtained by taking the derivative of the CCSD Lagrangian wrt. t1 and t2 amplitudes respectively.
            (Einstein summation):
            
            l1: <o|Hbar|phi^a_i>   + l_kc * <phi^c_k|Hbar|phi^a_i>   + l_klcd * <phi^cd_kl|Hbar|phi^a_i> = 0
            l2: <o|Hbar|phi^ab_ij> + l_kc * <phi^c_k|Hbar|phi^ab_ij> + l_klcd * <phi^cd_kl|Hbar|phi^ab_ij> = 0
            
            In Spin orbitals:
            l1:
            Hov_ia + l_ie * Hvv_ae - l_ma * Hoo_im + l_me * Hovvo_ieam + 0.5 * l_imef * Hvvvo_efam
            - 0.5 * l_mnae Hovoo_iemn - Gvv_ef * Hvovv_eifa - Goo_mn * Hooov_mina = 0
       
            where, Goo_mn = 0.5 * t_mjab * l_njab,  Gvv_ef = - 0.5 * l_ijeb * t_ijfb
            Intermediates Goo and Gvv have been built to bypass the construction of
            3-body Hbar terms like, l1  <-- Hvvooov_beilka * l_lkbe
                                    l1  <-- Hvvooov_cbijna * l_jncb
            
            l2:
             <ij||ab> + P(ab) l_ijae * Hov_eb - P(ij) l_imab * Hoo_jm + 0.5 * l_mnab * Hoooo_ijmn
             + 0.5 * Hvvvv_efab l_ijef + P(ij) l_ie * Hvovv_ejab - P(ab) l_ma * Hooov_ijmb
             + P(ij)P(ab) l_imae * Hovvo_jebm + P(ij)P(ab) l_ia * Hov_jb + P(ab) <ij||ae> * Gvv_be
             - P(ij) <im||ab> * Goo_mj
            
            Here we are using the unitary group approach (UGA) to derive spin adapted equations, please refer to chapter
            13 of reference 2 and notes in the current folder for more details. Lambda equations derived using UGA differ
            from the regular spin factorizd equations (PSI4) as follows:
            l_ia(UGA) = 2.0 * l_ia(PSI4)
            l_ijab(UGA) = 2.0 * (2.0 * l_ijab - l_ijba)
            The residual equations (without the preconditioner) follow the same relations as above.
            Ex. the inhomogenous terms in l1 and l2 equations in UGA are 2 * Hov_ia and 2.0 * (2.0 * <ij|ab> - <ij|ba>)
            respectively as opposed to Hov_ia and <ij|ab> in PSI4.
        """

        # l1 equations
        r_l1 = 2.0 * self.Hov.copy()
        r_l1 += ndot('ie,ea->ia', self.l1, self.Hvv)
        r_l1 -= ndot('im,ma->ia', self.Hoo, self.l1)
        r_l1 += ndot('ieam,me->ia', self.Hovvo, self.l1, prefactor=2.0)
        r_l1 += ndot('iema,me->ia', self.Hovov, self.l1, prefactor=-1.0)
        r_l1 += ndot('imef,efam->ia', self.l2, self.Hvvvo)
        r_l1 -= ndot('iemn,mnae->ia', self.Hovoo, self.l2)
        r_l1 -= ndot(
            'eifa,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=2.0)
        r_l1 -= ndot(
            'eiaf,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=-1.0)
        r_l1 -= ndot(
            'mina,mn->ia', self.Hooov, self.build_Goo(), prefactor=2.0)
        r_l1 -= ndot(
            'imna,mn->ia', self.Hooov, self.build_Goo(), prefactor=-1.0)

        # l2 equations
        # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
        r_l2 = self.Loovv.copy()
        r_l2 += ndot('ia,jb->ijab', self.l1, self.Hov, prefactor=2.0)
        r_l2 -= ndot('ja,ib->ijab', self.l1, self.Hov)
        r_l2 += ndot('ijeb,ea->ijab', self.l2, self.Hvv)
        r_l2 -= ndot('im,mjab->ijab', self.Hoo, self.l2)
        r_l2 += ndot('ijmn,mnab->ijab', self.Hoooo, self.l2, prefactor=0.5)
        r_l2 += ndot('ijef,efab->ijab', self.l2, self.Hvvvv, prefactor=0.5)
        r_l2 += ndot('ie,ejab->ijab', self.l1, self.Hvovv, prefactor=2.0)
        r_l2 += ndot('ie,ejba->ijab', self.l1, self.Hvovv, prefactor=-1.0)
        r_l2 -= ndot('mb,jima->ijab', self.l1, self.Hooov, prefactor=2.0)
        r_l2 -= ndot('mb,ijma->ijab', self.l1, self.Hooov, prefactor=-1.0)
        r_l2 += ndot('ieam,mjeb->ijab', self.Hovvo, self.l2, prefactor=2.0)
        r_l2 += ndot('iema,mjeb->ijab', self.Hovov, self.l2, prefactor=-1.0)
        r_l2 -= ndot('mibe,jema->ijab', self.l2, self.Hovov)
        r_l2 -= ndot('mieb,jeam->ijab', self.l2, self.Hovvo)
        r_l2 += ndot('ijeb,ae->ijab', self.Loovv, self.build_Gvv())
        r_l2 -= ndot('mi,mjab->ijab', self.build_Goo(), self.Loovv)

        old_l2 = self.l2.copy()
        old_l1 = self.l1.copy()

        # update l1 and l2 amplitudes
        self.l1 += r_l1 / self.Dia
        # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
        tmp = r_l2 / self.Dijab
        self.l2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

        # calculate rms from the residual
        rms = 2.0 * np.einsum('ia,ia->', old_l1 - self.l1, old_l1 - self.l1)
        rms += np.einsum('ijab,ijab->', old_l2 - self.l2, old_l2 - self.l2)
        return np.sqrt(rms)

    def pseudoenergy(self):
        pseudoenergy = 0
        pseudoenergy += ndot(
            'ijab,ijab->', self.get_MO('oovv'), self.l2, prefactor=0.5)
        return pseudoenergy

    def compute_lambda(self,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        cclambda_tstart = time.time()

        pseudoenergy_old = self.pseudoenergy()
        print("CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E" %
              (0, pseudoenergy_old, -pseudoenergy_old))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.l1, self.l2, max_diis)

        # Iterate!
        for CCLAMBDA_iter in range(1, maxiter + 1):

            rms = self.update()

            # Compute pseudoenergy
            pseudoenergy = self.pseudoenergy()

            # Print CCLAMBDA iteration information
            print(
                'CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E   DIIS = %d'
                % (CCLAMBDA_iter, pseudoenergy,
                   (pseudoenergy - pseudoenergy_old), diis_object.diis_size))

            # Check convergence
            if (rms < r_conv):
                print('\nCCLAMBDA has converged in %.3f seconds!' %
                      (time.time() - cclambda_tstart))
                return pseudoenergy

            # Update old pseudoenergy
            pseudoenergy_old = pseudoenergy

            #  Add the new error vector
            diis_object.add_error_vector(self.l1, self.l2)
            if CCLAMBDA_iter >= start_diis:
                self.l1, self.l2 = diis_object.extrapolate(self.l1, self.l2)


# End HelperCCLambda class
