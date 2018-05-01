"""
A simple python script to compute RHF-CCSD energy. Equations (Spin orbitals) from reference 1
have been spin-factored. However, explicit building of Wabef intermediates are avoided here.

References: 
1. J.F. Stanton, J. Gauss, J.D. Watts, and R.J. Bartlett, 
   J. Chem. Phys., volume 94, pp. 4334-4345 (1991).
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


class HelperCCEnergy(object):
    def __init__(self, mol, rhf_e, rhf_wfn, memory=2):

        print("\nInitalizing CCSD object...\n")

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.rhf_e = rhf_e
        self.wfn = rhf_wfn

        self.ccsd_corr_e = 0.0
        self.ccsd_e = 0.0

        self.ndocc = self.wfn.doccpi()[0]
        self.nmo = self.wfn.nmo()
        self.memory = memory
        self.C = self.wfn.Ca()
        self.npC = np.asarray(self.C)

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        H = np.asarray(self.mints.ao_kinetic()) + np.asarray(
            self.mints.ao_potential())
        self.nmo = H.shape[0]

        # Update H, transform to MO basis
        H = np.einsum('uj,vi,uv', self.npC, self.npC, H)

        print('Starting AO ->  MO transformation...')

        ERI_Size = self.nmo * 128.e-9
        memory_footprint = ERI_Size * 5
        if memory_footprint > self.memory:
            psi.clean()
            raise Exception(
                "Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint,
                                                   self.memory))

        # Integral generation from Psi4's MintsHelper
        self.MO = np.asarray(self.mints.mo_eri(self.C, self.C, self.C, self.C))
        # Physicist notation
        self.MO = self.MO.swapaxes(1, 2)
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." %
              (ERI_Size, self.nmo))

        # Update nocc and nvirt
        self.nocc = self.ndocc
        self.nvirt = self.nmo - self.nocc

        # Make slices
        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }

        # Compute Fock matrix
        self.F = H + 2.0 * np.einsum('pmqm->pq',
                                     self.MO[:, self.slice_o, :, self.slice_o])
        self.F -= np.einsum('pmmq->pq',
                            self.MO[:, self.slice_o, self.slice_o, :])

        ### Occupied and Virtual orbital energies
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]

        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(
            -1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        ### Construct initial guess
        print('Building initial guess...')
        # t^a_i
        self.t1 = np.zeros((self.nocc, self.nvirt))
        # t^{ab}_{ij}
        self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v,
                          self.slice_v] / self.Dijab

        print('\n..initialized CCSD in %.3f seconds.\n' %
              (time.time() - time_init))

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
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    #Equations from Reference 1 (Stanton's paper)

    #Bulid Eqn 9:
    def build_tilde_tau(self):
        ttau = self.t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        return ttau

    #Build Eqn 10:
    def build_tau(self):
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        return ttau

    #Build Eqn 3:
    def build_Fae(self):
        Fae = self.get_F('vv').copy()
        Fae -= ndot('me,ma->ae', self.get_F('ov'), self.t1, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', self.t1, self.get_MO('ovvv'), prefactor=2.0)
        Fae += ndot(
            'mf,maef->ae', self.t1, self.get_MO('ovvv'), prefactor=-1.0)
        Fae -= ndot(
            'mnaf,mnef->ae',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=2.0)
        Fae -= ndot(
            'mnaf,mnfe->ae',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=-1.0)
        return Fae

    #Build Eqn 4:
    def build_Fmi(self):
        Fmi = self.get_F('oo').copy()
        Fmi += ndot('ie,me->mi', self.t1, self.get_F('ov'), prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', self.t1, self.get_MO('ooov'), prefactor=2.0)
        Fmi += ndot(
            'ne,mnei->mi', self.t1, self.get_MO('oovo'), prefactor=-1.0)
        Fmi += ndot(
            'inef,mnef->mi',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=2.0)
        Fmi += ndot(
            'inef,mnfe->mi',
            self.build_tilde_tau(),
            self.get_MO('oovv'),
            prefactor=-1.0)
        return Fmi

    #Build Eqn 5:
    def build_Fme(self):
        Fme = self.get_F('ov').copy()
        Fme += ndot('nf,mnef->me', self.t1, self.get_MO('oovv'), prefactor=2.0)
        Fme += ndot(
            'nf,mnfe->me', self.t1, self.get_MO('oovv'), prefactor=-1.0)
        return Fme

    #Build Eqn 6:
    def build_Wmnij(self):
        Wmnij = self.get_MO('oooo').copy()
        Wmnij += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        Wmnij += ndot('ie,mnej->mnij', self.t1, self.get_MO('oovo'))
        # prefactor of 1 instead of 0.5 below to fold the last term of
        # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
        Wmnij += ndot(
            'ijef,mnef->mnij',
            self.build_tau(),
            self.get_MO('oovv'),
            prefactor=1.0)
        return Wmnij

    #Build Eqn 8:
    def build_Wmbej(self):
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1)
        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO('oovv'))
        Wmbej += ndot(
            'njfb,mnef->mbej', self.t2, self.get_MO('oovv'), prefactor=1.0)
        Wmbej += ndot(
            'njfb,mnfe->mbej', self.t2, self.get_MO('oovv'), prefactor=-0.5)
        return Wmbej

    # This intermediate appaears in the spin factorization of Wmbej terms.
    def build_Wmbje(self):
        Wmbje = -1.0 * (self.get_MO('ovov').copy())
        Wmbje -= ndot('jf,mbfe->mbje', self.t1, self.get_MO('ovvv'))
        Wmbje += ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1)
        Wmbje += ndot('jnfb,mnfe->mbje', tmp, self.get_MO('oovv'))
        return Wmbje

    # This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
    # as explicit construction of Wabef is avoided here.
    def build_Zmbij(self):
        Zmbij = 0
        Zmbij += ndot('mbef,ijef->mbij', self.get_MO('ovvv'), self.build_tau())
        return Zmbij

    def update(self):

        ### Build OEI intermediates
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F('ov').copy()
        r_T1 += ndot('ie,ae->ia', self.t1, Fae)
        r_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        r_T1 += ndot('imae,me->ia', self.t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', self.t2, Fme, prefactor=-1.0)
        r_T1 += ndot(
            'nf,nafi->ia', self.t1, self.get_MO('ovvo'), prefactor=2.0)
        r_T1 += ndot(
            'nf,naif->ia', self.t1, self.get_MO('ovov'), prefactor=-1.0)
        r_T1 += ndot(
            'mief,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=2.0)
        r_T1 += ndot(
            'mife,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=-1.0)
        r_T1 -= ndot(
            'mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=2.0)
        r_T1 -= ndot(
            'mnae,nmie->ia', self.t2, self.get_MO('ooov'), prefactor=-1.0)

        ### Build residual of T2 equations by spin adaptation of Eqn 2:
        # <ij||ab> ->  <ij|ab>
        #   spin   ->  spin-adapted (<alpha beta| alpha beta>)
        r_T2 = self.get_MO('oovv').copy()

        # Conventions used:
        #   P(ab) f(a,b) = f(a,b) - f(b,a)
        #   P(ij) f(i,j) = f(i,j) - f(j,i)
        #   P^(ab)_(ij) f(a,b,i,j) = f(a,b,i,j) + f(b,a,j,i)

        # P(ab) {t_ijae Fae_be}  ->  P^(ab)_(ij) {t_ijae Fae_be}
        tmp = ndot('ijae,be->ijab', self.t2, Fae)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        tmp = ndot('mb,me->be', self.t1, Fme)
        first = ndot('ijae,be->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
        tmp = ndot('imab,mj->ijab', self.t2, Fmi, prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
        tmp = ndot('je,me->jm', self.t1, Fme)
        first = ndot('imab,jm->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # Build TEI Intermediates
        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wmbej = self.build_Wmbej()
        Wmbje = self.build_Wmbje()
        Zmbij = self.build_Zmbij()

        # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
        # This also includes the last term in 0.5 * tau_ijef Wabef
        # as Wmnij is modified to include this contribution.
        r_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=1.0)

        # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
        # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
        # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
        # for in the contraction just above.

        # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
        r_T2 += ndot(
            'ijef,abef->ijab', tmp_tau, self.get_MO('vvvv'), prefactor=1.0)

        # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
        # where Zmbij_mbij = <mb|ef> * tau_ijef
        tmp = ndot('ma,mbij->ijab', self.t1, Zmbij)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
        # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imea,mbej->ijab', self.t2, Wmbej, prefactor=-1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
        tmp = ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        tmp += ndot('imae,mbje->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
        tmp = ndot('mjae,mbie->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
        #                                                      + t_ie * t_mb * <ma|je>}
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        tmp1 = ndot('imea,mbej->ijab', tmp, self.get_MO('ovvo'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        tmp = ndot('ie,mb->imeb', self.t1, self.t1)
        tmp1 = ndot('imeb,maje->ijab', tmp, self.get_MO('ovov'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
        tmp = ndot(
            'ie,abej->ijab', self.t1, self.get_MO('vvvo'), prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
        tmp = ndot(
            'ma,mbij->ijab', self.t1, self.get_MO('ovoo'), prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        ### Update T1 and T2 amplitudes
        self.t1 += r_T1 / self.Dia
        self.t2 += r_T2 / self.Dijab

        rms = np.einsum('ia,ia->', r_T1 / self.Dia, r_T1 / self.Dia)
        rms += np.einsum('ijab,ijab->', r_T2 / self.Dijab, r_T2 / self.Dijab)

        return np.sqrt(rms)

    def compute_corr_energy(self):
        CCSDcorr_E = 2.0 * np.einsum('ia,ia->', self.get_F('ov'), self.t1)
        tmp_tau = self.build_tau()
        CCSDcorr_E += 2.0 * np.einsum('ijab,ijab->', tmp_tau,
                                      self.get_MO('oovv'))
        CCSDcorr_E -= 1.0 * np.einsum('ijab,ijba->', tmp_tau,
                                      self.get_MO('oovv'))

        self.ccsd_corr_e = CCSDcorr_E
        self.ccsd_e = self.rhf_e + self.ccsd_corr_e
        return CCSDcorr_E

    def compute_energy(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        ccsd_tstart = time.time()

        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy()
        print(
            "CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" %
            (0, CCSDcorr_E_old, -CCSDcorr_E_old))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterate!
        for CCSD_iter in range(1, maxiter + 1):

            rms = self.update()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy()

            # Print CCSD iteration information
            print(
                'CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d'
                % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old),
                   diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                print('\nCCSD has converged in %.3f seconds!' %
                      (time.time() - ccsd_tstart))
                return CCSDcorr_E

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)


# End HelperCCEnergy class
