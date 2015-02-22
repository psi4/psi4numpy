# A SAPT helper object
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0
#

import numpy as np
import time


class helper_SAPT(object):

    def __init__(self, psi, energy, dimer, memory=2):
        print("\nInitalizing SAPT object...\n")
        tinit_start = time.time()

        # Set a few crucial attributes
        dimer.reset_point_group('c1')
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi.clean()
            raise Exception("Found %d fragments, must be 2." % nfrags)

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name('monomerA')
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name('monomerB')

        # Compute monomer properties
        tstart = time.time()
        psi.set_active_molecule(monomerA)
        self.V_A = np.asanyarray(psi.MintsHelper().ao_potential())
        self.rhfA = energy('RHF')
        self.wfnA = psi.wavefunction()
        print("RHF for monomer A finished in %.2f seconds." % (time.time() - tstart))

        tstart = time.time()
        psi.set_active_molecule(monomerB)
        self.V_B = np.asanyarray(psi.MintsHelper().ao_potential())
        self.rhfB = energy('RHF')
        self.wfnB = psi.wavefunction()
        print("RHF for monomer B finished in %.2f seconds." % (time.time() - tstart))

        # Setup a few variables
        self.psi = psi
        self.memory = memory
        self.nmo = self.wfnA.nmo()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        self.idx_A = ['a', 'r']

        self.C_A = np.asanyarray(self.wfnA.Ca())
        self.Co_A = self.wfnA.Ca_subset("AO", "ACTIVE_OCC")
        self.Cv_A = self.wfnA.Ca_subset("AO", "ACTIVE_VIR")
        self.eps_A = np.asanyarray(self.wfnA.epsilon_a())

        # Monomer B
        self.nuc_rep_B = monomerB.nuclear_repulsion_energy()
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B
        self.idx_B = ['b', 's']

        self.C_B = np.asanyarray(self.wfnB.Ca())
        self.Co_B = self.wfnB.Ca_subset("AO", "ACTIVE_OCC")
        self.Cv_B = self.wfnB.Ca_subset("AO", "ACTIVE_VIR")
        self.eps_B = np.asanyarray(self.wfnB.epsilon_a())

        # Dimer
        self.nuc_rep = dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B
        self.vt_nuc_rep = self.nuc_rep / (4 * self.ndocc_A * self.ndocc_B)

        # Make slice, orbital, and size dictionaries
        self.slices = {'a': slice(0, self.ndocc_A),
                       'r': slice(self.ndocc_A, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(self.ndocc_B, None)}

        self.orbitals = {'a': self.Co_A,
                         'r': self.Cv_A,
                         'b': self.Co_B,
                         's': self.Cv_B}

        self.sizes = {'a': self.ndocc_A,
                      'r': self.nvirt_A,
                      'b': self.ndocc_B,
                      's': self.nvirt_B}

        # Compute size of ERI tensor in GB
        psi.set_active_molecule(dimer)
        mints = psi.MintsHelper()
        self.mints = mints
        ERI_Size = (self.nmo ** 4) * 8.e-9
        memory_footprint = ERI_Size * 5
        if memory_footprint > self.memory:
            self.psi.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, numpy_memory))

        # Integral generation from Psi4's MintsHelper
        print('Building ERI tensor...')
        tstart = time.time()
        # Leave ERI as a Psi4 Matrix
        self.I = mints.ao_eri()
        print('...built ERI tensor in %.3f seconds.' % (time.time() - tstart))
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))
        self.S = np.asanyarray(mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = np.einsum('ui,vj,uv->ij', self.C_B, self.C_B, self.V_A)
        self.V_A_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_A)
        self.V_B_AA = np.einsum('ui,vj,uv->ij', self.C_A, self.C_A, self.V_B)
        self.V_B_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_B)

        self.S_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.S)

        print("\n...finished inializing SAPT object in %5.2f seconds." % (time.time() - tinit_start))

    # Compute MO ERI tensor (v) on the fly
    def v(self, string):
        if len(string) != 4:
            self.psi.clean()
            raise Exception('v: string %s does not have 4 elements' % string)

        # ERI's from mints are of type (11|22) - need <12|12>
        orbitals = [self.orbitals[string[0]], self.orbitals[string[2]],
                    self.orbitals[string[1]], self.orbitals[string[3]]]
        v = self.mints.mo_transform(self.I, *orbitals)
        return np.asanyarray(v).swapaxes(1, 2)

    # Grab MO overlap matrices
    def s(self, string):
        if len(string) != 2:
            self.psi.clean()
            raise Exception('S: string %s does not have 2 elements.' % string)

        s1 = string[0]
        s2 = string[1]

        # Compute on the fly
        # return np.einsum('ui,vj,uv->ij', self.orbitals[string[0]], self.orbitals[string[1]], self.S)

        # Same monomer and index- return diaganol
        if (s1 == s2):
            return np.diag(np.ones(self.sizes[s1]))

        # Same monomer, but O-V or V-O means zeros array
        elif (s1 in self.idx_A) and (s2 in self.idx_A):
            return np.zeros((self.sizes[s1], self.sizes[s2]))
        elif (s1 in self.idx_B) and (s2 in self.idx_B):
            return np.zeros((self.sizes[s1], self.sizes[s2]))

        # Return S_AB
        elif (s1 in self.idx_B):
            return self.S_AB[self.slices[s2], self.slices[s1]].T
        else:
            return self.S_AB[self.slices[s1], self.slices[s2]]

    # Grab epsilons, reshape if requested
    def eps(self, string, dim=1):
        if len(string) != 1:
            self.psi.clean()
            raise Exception('Epsilon: string %s does not have 1 element.' % string)

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == 'b') or (string == 's'):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            return self.eps_A[self.slices[string]].reshape(shape)

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
            self.psi.clean()
            raise Exception('Potential: string %s does not have 2 elements.' % string)

        s1 = string[0]
        s2 = string[1]

        # Two seperate cases
        if side == 'A':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_A) / (2 * self.ndocc_A)

            if (s1 in self.idx_B) and (s2 in self.idx_B):
                return self.V_A_BB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_A)
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_A_AB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_A)
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_A_AB[self.slices[s2], self.slices[s1]].T / (2 * self.ndocc_A)
            else:
                self.psi.clean()
                raise Exception('No match for %s indices in helper_SAPT.potential.' % string)

        elif side == 'B':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_B) / (2 * self.ndocc_B)

            if (s1 in self.idx_A) and (s2 in self.idx_A):
                return self.V_B_AA[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_B)
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_B_AB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_B)
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_B_AB[self.slices[s2], self.slices[s1]].T / (2 * self.ndocc_B)
            else:
                self.psi.clean()
                raise Exception('No match for %s indices in helper_SAPT.potential.' % string)
        else:
            self.psi.clean()
            raise Exception('helper_SAPT.potential side must be either A or B, not %s.' % side)

    # Compute V tilde, Index as V_{1,2}^{3,4}
    def vt(self, string):
        if len(string) != 4:
            self.psi.clean()
            raise Exception('Compute tilde{V}: string %s does not have 4 elements' % string)

        # Grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        V = self.v(string)

        # Potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A')
        V += np.einsum('ik,jl->ijkl', S_A, V_A)

        # Potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B')
        V += np.einsum('ik,jl->ijkl', V_B, S_B)

        # Nuclear
        V += np.einsum('ik,jl->ijkl', S_A, S_B) * self.vt_nuc_rep

        return V

    # Compute CPHF orbitals
    def chf(self, monomer, ind=False):
        if monomer not in ['A', 'B']:
            self.psi.clean()
            raise Exception('%s is not a valid monomer for CHF.' % monomer)

        if monomer == 'A':
            # Form electostatic potential
            w_n = 2 * np.einsum('saba->bs', self.v('saba'))
            w_n += self.V_A_BB[self.slices['b'], self.slices['s']]
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))

            # Set terms
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            v_term3 = 'ssbb'
            no, nv = self.ndocc_B, self.nvirt_B

        if monomer == 'B':
            w_n = 2 * np.einsum('rbab->ar', self.v('rbab'))
            w_n += self.V_B_AA[self.slices['a'], self.slices['r']]
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            v_term3 = 'rraa'
            no, nv = self.ndocc_A, self.nvirt_A

        # Form A matix (LHS)
        v_vOoV = 2 * self.v(v_term1) - self.v(v_term2).swapaxes(2, 3)
        v_ooaa = self.v(v_term3)
        v_vVoO = 2 * v_ooaa - v_ooaa.swapaxes(2, 3)
        A_ovOV = np.einsum('vOoV->ovOV', v_vOoV + v_vVoO.swapaxes(1, 3))

        # Mangled the indices so badly with strides we need to copy back to C contigous
        nov = nv * no
        A_ovOV = A_ovOV.reshape(nov, nov).copy(order='C')
        A_ovOV[np.diag_indices_from(A_ovOV)] -= eps_ov.ravel()

        # Call DGESV, need flat ov array
        B_ov = -1 * w_n.ravel()
        t = np.linalg.solve(A_ovOV, B_ov)
        # Our notation wants vo array
        t = t.reshape(no, nv).T

        if ind:
            # E200 Induction energy is free at the point
            e20_ind = 2 * np.einsum('vo,ov->', t, w_n)
            return (t, e20_ind)
        else:
            return t
# End SAPT helper


class sapt_timer(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        print('\nStarting %s...' % name)

    def stop(self):
        t = time.time() - self.start
        print('...%s took a total of % .2f seconds.' % (self.name, t))


def sapt_printer(line, value):
    spacer = ' ' * (20 - len(line))

    print(line + spacer + '% 16.8f mH  % 16.8f kcal/mol' % (value * 1000, value * 627.509))
