"""Helper classes and functions for spin-orbital CC module.

References:
- DPD formulation of CCSD equations: [Stanton:1991:4334]
- CC algorithms from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-02-22"

import time
import numpy as np
import psi4


# N dimensional dot
# Like a mini DPD library
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s), )
        right_pos += (input_right.find(s), )
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed), op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T, op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)


class helper_CCSD(object):
    def __init__(self, mol, freeze_core=False, memory=2):
        """
        Initializes the helper_CCSD object.

        Parameters
        ----------
        mol : psi4.core.Molecule
            The molecule to be used for the given helper object.
        freeze_core : {True, False}, optional
            Boolean flag indicating presence of frozen core orbitals.  Default: False
        memory : int or float, optional
            The total memory, in GB, allotted for the given helper object. Default: 2

        Examples
        --------

        # Construct the helper object
        >>> ccsd = helper_CCSD(psi4.geometry("He\nHe 1 2.0"))

        # Compute the CCSD total energy
        >>> total_E = ccsd.compute_energy()
        """

        if freeze_core:
            raise Exception("Frozen core doesnt work yet!")
        print("\nInitalizing CCSD object...\n")

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        print('Computing RHF reference.')
        psi4.core.set_active_molecule(mol)
        psi4.set_module_options('SCF', {'SCF_TYPE': 'PK'})
        psi4.set_module_options('SCF', {'E_CONVERGENCE': 10e-10})
        psi4.set_module_options('SCF', {'D_CONVERGENCE': 10e-10})

        # Core is frozen by default
        if not freeze_core:
            psi4.set_module_options('CCENERGY', {'FREEZE_CORE': 'FALSE'})

        self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)
        print('RHF Final Energy                          % 16.10f\n' % (self.rhf_e))

        self.ccsd_corr_e = 0.0
        self.ccsd_e = 0.0

        self.eps = np.asarray(self.wfn.epsilon_a())
        self.ndocc = self.wfn.doccpi()[0]
        self.nmo = self.wfn.nmo()
        self.memory = memory
        self.nfzc = 0

        # Freeze core
        if freeze_core:
            Zlist = np.array([mol.Z(x) for x in range(mol.natom())])
            self.nfzc = np.sum(Zlist > 2)
            self.nfzc += np.sum(Zlist > 10) * 4
            if np.any(Zlist > 18):
                raise Exception("Frozen core for Z > 18 not yet implemented")

            print("Cutting %d core orbitals." % self.nfzc)

            # Copy C
            oldC = np.array(self.wfn.Ca(), copy=True)

            # Build new C matrix and view, set with numpy slicing
            self.C = psi.Matrix(self.nmo, self.nmo - self.nfzc)
            self.npC = np.asarray(self.C)
            self.npC[:] = oldC[:, self.nfzc:]

            # Update epsilon array
            self.ndocc -= self.nfzc

        else:
            self.C = self.wfn.Ca()
            self.npC = np.asarray(self.C)

        mints = psi4.core.MintsHelper(self.wfn.basisset())
        H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        self.nmo = H.shape[0]

        # Update H, transform to MO basis and tile for alpha/beta spin
        H = np.einsum('uj,vi,uv', self.npC, self.npC, H)
        H = np.repeat(H, 2, axis=0)
        H = np.repeat(H, 2, axis=1)

        # Make H block diagonal
        spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
        H *= (spin_ind.reshape(-1, 1) == spin_ind)

        #Make spin-orbital MO
        print('Starting AO -> spin-orbital MO transformation...')

        ERI_Size = (self.nmo**4) * 128.e-9
        memory_footprint = ERI_Size * 5
        if memory_footprint > self.memory:
            psi.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, self.memory))

        # Integral generation from Psi4's MintsHelper
        self.MO = np.asarray(mints.mo_spin_eri(self.C, self.C))
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))

        # Update nocc and nvirt
        self.nso = self.nmo * 2
        self.nfzc = self.nfzc * 2
        self.nocc = self.ndocc * 2
        self.nvirt = self.nso - self.nocc - self.nfzc * 2

        # Make slices
        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nso)
        self.slice_a = slice(0, self.nso)
        self.slice_dict = {'f': self.slice_nfzc, 'o': self.slice_o, 'v': self.slice_v, 'a': self.slice_a}

        # Extend eigenvalues
        self.eps = np.repeat(self.eps, 2)

        # Compute Fock matrix
        self.F = H + np.einsum('pmqm->pq', self.MO[:, self.slice_o, :, self.slice_o])

        ### Build D matrices
        print('\nBuilding denominator arrays...')
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]

        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        ### Construct initial guess
        print('Building initial guess...')
        # t^a_i
        self.t1 = np.zeros((self.nocc, self.nvirt))
        # t^{ab}_{ij}
        self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / self.Dijab

        print('\n..initialed CCSD in %.3f seconds.\n' % (time.time() - time_init))

    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v
    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]], self.slice_dict[string[2]],
                       self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    #Bulid Eqn 9: tilde{\Tau})
    def build_tilde_tau(self):
        """Builds [Stanton:1991:4334] Eqn. 9"""
        ttau = self.t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    #Build Eqn 10: \Tau)
    def build_tau(self):
        """Builds [Stanton:1991:4334] Eqn. 10"""
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    #Build Eqn 3:
    def build_Fae(self):
        """Builds [Stanton:1991:4334] Eqn. 10"""
        Fae = self.get_F('vv').copy()
        Fae[np.diag_indices_from(Fae)] = 0

        Fae -= ndot('me,ma->ae', self.get_F('ov'), self.t1, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', self.t1, self.get_MO('ovvv'))

        Fae -= ndot('mnaf,mnef->ae', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=0.5)
        return Fae

    #Build Eqn 4:
    def build_Fmi(self):
        """Builds [Stanton:1991:4334] Eqn. 4"""
        Fmi = self.get_F('oo').copy()
        Fmi[np.diag_indices_from(Fmi)] = 0

        Fmi += ndot('ie,me->mi', self.t1, self.get_F('ov'), prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', self.t1, self.get_MO('ooov'))

        Fmi += ndot('inef,mnef->mi', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=0.5)
        return Fmi

    #Build Eqn 5:
    def build_Fme(self):
        """Builds [Stanton:1991:4334] Eqn. 5"""
        Fme = self.get_F('ov').copy()
        Fme += ndot('nf,mnef->me', self.t1, self.get_MO('oovv'))
        return Fme

    #Build Eqn 6:
    def build_Wmnij(self):
        """Builds [Stanton:1991:4334] Eqn. 6"""
        Wmnij = self.get_MO('oooo').copy()

        Pij = ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        Wmnij += Pij
        Wmnij -= Pij.swapaxes(2, 3)

        Wmnij += ndot('ijef,mnef->mnij', self.build_tau(), self.get_MO('oovv'), prefactor=0.25)
        return Wmnij

    #Build Eqn 7:
    def build_Wabef(self):
        """Builds [Stanton:1991:4334] Eqn. 7"""
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper
        Wabef = self.get_MO('vvvv').copy()

        Pab = ndot('mb,amef->abef', self.t1, self.get_MO('vovv'))
        Wabef -= Pab
        Wabef += Pab.swapaxes(0, 1)

        Wabef += ndot('mnab,mnef->abef', self.build_tau(), self.get_MO('oovv'), prefactor=0.25)
        return Wabef

    #Build Eqn 8:
    def build_Wmbej(self):
        """Builds [Stanton:1991:4334] Eqn. 8"""
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))

        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1)

        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO('oovv'))
        return Wmbej

    def update(self):
        """Updates T1 & T2 amplitudes."""

        ### Build intermediates: [Stanton:1991:4334] Eqns. 3-8
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build RHS side of self.t1 equations, [Stanton:1991:4334] Eqn. 1
        rhs_T1 = self.get_F('ov').copy()
        rhs_T1 += ndot('ie,ae->ia', self.t1, Fae)
        rhs_T1 -= ndot('ma,mi->ia', self.t1, Fmi)
        rhs_T1 += ndot('imae,me->ia', self.t2, Fme)
        rhs_T1 -= ndot('nf,naif->ia', self.t1, self.get_MO('ovov'))
        rhs_T1 -= ndot('imef,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=0.5)
        rhs_T1 -= ndot('mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=0.5)

        ### Build RHS side of self.t2 equations, [Stanton:1991:4334] Eqn. 2
        rhs_T2 = self.get_MO('oovv').copy()

        # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
        tmp = Fae - 0.5 * ndot('mb,me->be', self.t1, Fme)
        Pab = ndot('ijae,be->ijab', self.t2, tmp)
        rhs_T2 += Pab
        rhs_T2 -= Pab.swapaxes(2, 3)

        # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
        tmp = Fmi + 0.5 * ndot('je,me->mj', self.t1, Fme)
        Pij = ndot('imab,mj->ijab', self.t2, tmp)
        rhs_T2 -= Pij
        rhs_T2 += Pij.swapaxes(0, 1)

        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wabef = self.build_Wabef()
        rhs_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=0.5)
        rhs_T2 += ndot('ijef,abef->ijab', tmp_tau, Wabef, prefactor=0.5)

        # P_(ij) * P_(ab)
        # (ij - ji) * (ab - ba)
        # ijab - ijba -jiab + jiba
        tmp = ndot('ie,mbej->mbij', self.t1, self.get_MO('ovvo'))
        tmp = ndot('ma,mbij->ijab', self.t1, tmp)
        Wmbej = self.build_Wmbej()
        Pijab = ndot('imae,mbej->ijab', self.t2, Wmbej) - tmp

        rhs_T2 += Pijab
        rhs_T2 -= Pijab.swapaxes(2, 3)
        rhs_T2 -= Pijab.swapaxes(0, 1)
        rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

        Pij = ndot('ie,abej->ijab', self.t1, self.get_MO('vvvo'))
        rhs_T2 += Pij
        rhs_T2 -= Pij.swapaxes(0, 1)

        Pab = ndot('ma,mbij->ijab', self.t1, self.get_MO('ovoo'))
        rhs_T2 -= Pab
        rhs_T2 += Pab.swapaxes(2, 3)

        ### Update T1 and T2 amplitudes
        self.t1 = rhs_T1 / self.Dia
        self.t2 = rhs_T2 / self.Dijab

    def compute_corr_energy(self):
        """Compute CCSD correlation energy using current amplitudes."""
        CCSDcorr_E = np.einsum('ia,ia->', self.get_F('ov'), self.t1)
        CCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', self.get_MO('oovv'), self.t2)
        CCSDcorr_E += 0.5 * np.einsum('ijab,ia,jb->', self.get_MO('oovv'), self.t1, self.t1)

        self.ccsd_corr_e = CCSDcorr_E
        self.ccsd_e = self.rhf_e + self.ccsd_corr_e
        return CCSDcorr_E

    def compute_energy(self, e_conv=1.e-8, maxiter=20, max_diis=8):
        """Computes total CCSD energy."""
        ### Setup DIIS
        diis_vals_t1 = [self.t1.copy()]
        diis_vals_t2 = [self.t2.copy()]
        diis_errors = []

        ### Start Iterations
        ccsd_tstart = time.time()

        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy()
        print("CCSD Iteration %3d: CCSD correlation = %.12f   dE = % .5E   MP2" % (0, CCSDcorr_E_old, -CCSDcorr_E_old))

        # Iterate!
        diis_size = 0
        for CCSD_iter in range(1, maxiter + 1):

            # Save new amplitudes
            oldt1 = self.t1.copy()
            oldt2 = self.t2.copy()

            self.update()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy()

            # Print CCSD iteration information
            print('CCSD Iteration %3d: CCSD correlation = %.12f   dE = % .5E   DIIS = %d' %
                  (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old), diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv):
                print('\nCCSD has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return CCSDcorr_E

            # Add DIIS vectors
            diis_vals_t1.append(self.t1.copy())
            diis_vals_t2.append(self.t2.copy())

            # Build new error vector
            error_t1 = (diis_vals_t1[-1] - oldt1).ravel()
            error_t2 = (diis_vals_t2[-1] - oldt2).ravel()
            diis_errors.append(np.concatenate((error_t1, error_t2)))

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            if CCSD_iter >= 1:
                # Limit size of DIIS vector
                if (len(diis_vals_t1) > max_diis):
                    del diis_vals_t1[0]
                    del diis_vals_t2[0]
                    del diis_errors[0]

                diis_size = len(diis_vals_t1) - 1

                # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
                B = np.ones((diis_size + 1, diis_size + 1)) * -1
                B[-1, -1] = 0

                for n1, e1 in enumerate(diis_errors):
                    B[n1, n1] = np.dot(e1, e1)
                    for n2, e2 in enumerate(diis_errors):
                        if n1 >= n2: continue
                        B[n1, n2] = np.dot(e1, e2)
                        B[n2, n1] = B[n1, n2]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
                resid = np.zeros(diis_size + 1)
                resid[-1] = -1

                # Solve pulay equations, [Pulay:1980:393], Eqn. 6
                ci = np.linalg.solve(B, resid)

                # Calculate new amplitudes
                self.t1[:] = 0
                self.t2[:] = 0
                for num in range(diis_size):
                    self.t1 += ci[num] * diis_vals_t1[num + 1]
                    self.t2 += ci[num] * diis_vals_t2[num + 1]

            # End DIIS amplitude update
            # End CCSD class


if __name__ == "__main__":
    arr4 = np.random.rand(4, 4, 4, 4)
    arr2 = np.random.rand(4, 4)

    def test_ndot(string, op1, op2):
        ein_ret = np.einsum(string, op1, op2)
        ndot_ret = ndot(string, op1, op2)
        assert np.allclose(ein_ret, ndot_ret)

    test_ndot('abcd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abfe', arr4, arr4)
    test_ndot('mnab,mnij->ijab', arr4, arr4)

    test_ndot('cd,cdef->ef', arr2, arr4)
    test_ndot('ce,cdef->df', arr2, arr4)
    test_ndot('nf,naif->ia', arr2, arr4)
