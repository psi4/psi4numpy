# A SAPT helper object
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0
#

import numpy as np
import time
import psi4

class helper_SAPT(object):

    def __init__(self, dimer, memory=8, algorithm='MO', reference='RHF'):
        print("\nInitializing SAPT object...\n")
        tinit_start = time.time()

        # Set a few crucial attributes
        self.alg = algorithm.upper()
        self.reference = reference.upper()
        dimer.reset_point_group('c1')
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi4.core.clean()
            raise Exception("Found %d fragments, must be 2." % nfrags)

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name('monomerA')
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name('monomerB')
        self.mult_A = monomerA.multiplicity()
        self.mult_B = monomerB.multiplicity()

        # Compute monomer properties

        tstart = time.time()
        self.rhfA, self.wfnA = psi4.energy('SCF', return_wfn=True, molecule=monomerA)
        self.V_A = np.asarray(psi4.core.MintsHelper(self.wfnA.basisset()).ao_potential())
        print("RHF for monomer A finished in %.2f seconds." % (time.time() - tstart))

        tstart = time.time()
        self.rhfB, self.wfnB = psi4.energy('SCF', return_wfn=True, molecule=monomerB)
        self.V_B = np.asarray(psi4.core.MintsHelper(self.wfnB.basisset()).ao_potential())
        print("RHF for monomer B finished in %.2f seconds." % (time.time() - tstart))

        # Setup a few variables
        self.memory = memory
        self.nmo = self.wfnA.nmo()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        if reference == 'ROHF':
          self.idx_A = ['i', 'a', 'r']
          self.nsocc_A = self.wfnA.soccpi()[0]
          occA = self.ndocc_A + self.nsocc_A
        else:
          self.idx_A = ['a', 'r']
          self.nsocc_A = 0
          occA = self.ndocc_A 

        self.C_A = np.asarray(self.wfnA.Ca())
        self.Co_A = self.C_A[:, :self.ndocc_A]
        self.Ca_A = self.C_A[:, self.ndocc_A:occA]
        self.Cv_A = self.C_A[:, occA:]
        self.eps_A = np.asarray(self.wfnA.epsilon_a())

        # Monomer B
        self.nuc_rep_B = monomerB.nuclear_repulsion_energy()
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B
        if reference == 'ROHF':
          self.idx_B = ['j', 'b', 's']
          self.nsocc_B = self.wfnB.soccpi()[0]
          occB = self.ndocc_B + self.nsocc_B
        else:
          self.idx_B = ['b', 's']
          self.nsocc_B = 0
          occB = self.ndocc_B 

        self.C_B = np.asarray(self.wfnB.Ca())
        self.Co_B = self.C_B[:, :self.ndocc_B]
        self.Ca_B = self.C_B[:, self.ndocc_B:occB]
        self.Cv_B = self.C_B[:, occB:]
        self.eps_B = np.asarray(self.wfnB.epsilon_a())

        # Dimer
        self.nuc_rep = dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B
        self.vt_nuc_rep = self.nuc_rep / ((2 * self.ndocc_A + self.nsocc_A)
                                           * (2 * self.ndocc_B + self.nsocc_B))

        # Make slice, orbital, and size dictionaries
        if reference == 'ROHF':
          self.slices = {
                       'i': slice(0, self.ndocc_A),
                       'a': slice(self.ndocc_A, occA),
                       'r': slice(occA, None),
                       'j': slice(0, self.ndocc_B),
                       'b': slice(self.ndocc_B, occB),
                       's': slice(occB, None)
                      }

          self.orbitals = {'i': self.Co_A,
                           'a': self.Ca_A,
                           'r': self.Cv_A,
                           'j': self.Co_B,
                           'b': self.Ca_B,
                           's': self.Cv_B
                        }

          self.sizes = {'i': self.ndocc_A,
                        'a': self.nsocc_A,
                        'r': self.nvirt_A,
                        'j': self.ndocc_B,
                        'b': self.nsocc_B,
                        's': self.nvirt_B}

        else:
          self.slices = {
                       'a': slice(0, self.ndocc_A),
                       'r': slice(occA, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(occB, None)
                      }

          self.orbitals = {'a': self.Co_A,
                           'r': self.Cv_A,
                           'b': self.Co_B,
                           's': self.Cv_B
                        }

          self.sizes = {'a': self.ndocc_A,
                        'r': self.nvirt_A,
                        'b': self.ndocc_B,
                        's': self.nvirt_B}

        # Compute size of ERI tensor in GB
        self.dimer_wfn = psi4.core.Wavefunction.build(dimer, psi4.core.get_global_option('BASIS'))
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints
        ERI_Size = (self.nmo ** 4) * 8.e-9
        memory_footprint = ERI_Size * 4
        if memory_footprint > self.memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, self.memory))

        # Integral generation from Psi4's MintsHelper
        print('Building ERI tensor...')
        tstart = time.time()
        # Leave ERI as a Psi4 Matrix
        self.I = np.asarray(self.mints.ao_eri()).swapaxes(1,2)
        print('...built ERI tensor in %.3f seconds.' % (time.time() - tstart))
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))
        self.S = np.asarray(self.mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = np.einsum('ui,vj,uv->ij', self.C_B, self.C_B, self.V_A)
        self.V_A_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_A)
        self.V_B_AA = np.einsum('ui,vj,uv->ij', self.C_A, self.C_A, self.V_B)
        self.V_B_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_B)

        self.S_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.S)

        if self.alg == "AO":
            tstart = time.time()
            aux_basis = psi4.core.BasisSet.build(self.dimer_wfn.molecule(), "DF_BASIS_SCF",
                                            psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                            "JKFIT", psi4.core.get_global_option('BASIS'),
                                            puream=self.dimer_wfn.basisset().has_puream())

            self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
            self.jk.set_memory(int(memory * 1e9))
            self.jk.initialize()
            print("\n...initialized JK objects in %5.2f seconds." % (time.time() - tstart))

        print("\n...finished initializing SAPT object in %5.2f seconds." % (time.time() - tinit_start))

    # Compute MO ERI tensor (v) on the fly
    def v(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('v: string %s does not have 4 elements' % string)

        # ERI's from mints are of type (11|22) - need <12|12>
        V = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], self.I)
        V = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], V)
        V = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], V)
        V = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], V)
        return V

    # Grab MO overlap matrices
    def s(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('S: string %s does not have 2 elements.' % string)

        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1,1)

        s1 = string[0]
        s2 = string[1]

        # Compute on the fly
        return (self.orbitals[string[0]].T).dot(self.S).dot(self.orbitals[string[1]])
        #return np.einsum('ui,vj,uv->ij', self.orbitals[string[0]], self.orbitals[string[1]], self.S)

    # Grab epsilons, reshape if requested
    def eps(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise Exception('Epsilon: string %s does not have 1 element.' % string)

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == 'i') or (string == 'a') or (string == 'r'):
            return self.eps_A[self.slices[string]].reshape(shape)
        elif (string == 'j') or (string == 'b') or (string == 's'):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            psi4.core.clean()
            raise Exception('Unknown orbital type in eps: %s.' % string)

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('Potential: string %s does not have 2 elements.' % string)

        s1 = string[0]
        s2 = string[1]

        # Two separate cases
        if side == 'A':
            # Compute on the fly
            return (self.orbitals[string[0]].T).dot(self.V_A).dot(self.orbitals[string[1]])
            #return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_A)

        elif side == 'B':
            # Compute on the fly
            return (self.orbitals[string[0]].T).dot(self.V_B).dot(self.orbitals[string[1]])
            #return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_B)
        else:
            psi4.core.clean()
            raise Exception('helper_SAPT.potential side must be either A or B, not %s.' % side)

    # Compute V tilde, Index as V_{1,2}^{3,4}
    def vt(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('Compute tilde{V}: string %s does not have 4 elements' % string)

        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1,1,1,1)

        # Grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        V = self.v(string)
        # Potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A') / (2 * self.ndocc_A + self.nsocc_A)
        V += np.einsum('ik,jl->ijkl', S_A, V_A)

        # Potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B') / (2 * self.ndocc_B + self.nsocc_B)
        #print s_right, np.abs(V_B).sum()
        V += np.einsum('ik,jl->ijkl', V_B, S_B)

        # Nuclear
        V += np.einsum('ik,jl->ijkl', S_A, S_B) * self.vt_nuc_rep

        return V

    # Compute CPHF orbitals
    def chf(self, monomer, ind=False):
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception('%s is not a valid monomer for CHF.' % monomer)

        if self.reference == 'ROHF':
            psi4.core.clean()
            raise Exception('CPHF for a ROHF reference not implemented yet.')

        if monomer == 'A':
            # Form electrostatic potential
            w_n = 2 * np.einsum('saba->bs', self.v('saba'))
            w_n += self.V_A_BB[self.slices['b'], self.slices['s']]
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))

            # Set terms
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            no, nv = self.ndocc_B, self.nvirt_B

        if monomer == 'B':
            w_n = 2 * np.einsum('rbab->ar', self.v('rbab'))
            w_n += self.V_B_AA[self.slices['a'], self.slices['r']]
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            no, nv = self.ndocc_A, self.nvirt_A

        # Form A matrix (LHS)
        voov = self.v(v_term1)
        v_vOoV = 2 * voov - self.v(v_term2).swapaxes(2, 3)
        v_ooaa = voov.swapaxes(1, 3)
        v_vVoO = 2 * v_ooaa - v_ooaa.swapaxes(2, 3)
        A_ovOV = np.einsum('vOoV->ovOV', v_vOoV + v_vVoO.swapaxes(1, 3))

        # Mangled the indices so badly with strides we need to copy back to C contiguous
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

    def compute_sapt_JK(self, Cleft, Cright, tensor=None):

        if self.alg != "AO":
            raise Exception("Attempted a call to JK builder in an MO algorithm")

        if self.reference == "ROHF":
            raise Exception("AO algorithm not yet implemented for ROHF reference.")

        return_single = False
        if not isinstance(Cleft, (list, tuple)):
            Cleft = [Cleft]
            return_single = True
        if not isinstance(Cright, (list, tuple)):
            Cright = [Cright]
            return_single = True
        if (not isinstance(tensor, (list, tuple))) and (tensor is not None):
            tensor = [tensor]
            return_single = True

        if len(Cleft) != len(Cright):
            raise Exception("Cleft list is not the same length as Cright list")

        zero_append = []
        num_compute = 0

        for num in range(len(Cleft)):
            Cl = Cleft[num]
            Cr = Cright[num]

            if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
                zero_append.append(num)
                continue

            if tensor is not None:
                mol = Cl.shape[1]
                mor = Cr.shape[1]
                
                if (tensor[num].shape[0] != mol) or (tensor[num].shape[1] != mor):
                    raise Exception("compute_sapt_JK: Tensor size does not match Cl (%d) /Cr (%d) : %s" %
                                                            (mol, mor, str(tensor[num].shape)))
                if mol < mor:
                    Cl = np.dot(Cl, tensor[num])
                else:
                    Cr = np.dot(Cr, tensor[num].T)

            Cl = psi4.core.Matrix.from_array(Cl)
            Cr = psi4.core.Matrix.from_array(Cr)

            self.jk.C_left_add(Cl)
            self.jk.C_right_add(Cr)
            num_compute += 1
        
        self.jk.compute() 

        J_list = []
        K_list = []
        for num in range(num_compute):
            J_list.append(np.array(self.jk.J()[num])) 
            K_list.append(np.array(self.jk.K()[num])) 

        self.jk.C_clear()

        z = np.zeros((self.nmo, self.nmo))
        for num in zero_append:
            J_list.insert(num, z)
            K_list.insert(num, z)

        if return_single:
            return J_list[0], K_list[0]
        else:
            return J_list, K_list

    def chain_dot(self, *dot_list):
        result = dot_list[0]
        for x in range(len(dot_list) - 1):
            result = np.dot(result, dot_list[x + 1])
        return result

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
# End SAPT helper
