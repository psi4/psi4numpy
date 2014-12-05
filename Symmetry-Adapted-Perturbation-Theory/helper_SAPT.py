import numpy as np
import time


class helper_SAPT(object):
    
    def __init__(self, psi, energy, dimer, memory=2):
        print("\nInializing SAPT object...\n")
        t = time.time()

        # Set a few crucial attributes
        dimer.reset_point_group('c1')
        dimer.fix_orientation(True)
        dimer.fix_com(True) 
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags!=2:
            clean()
            raise Exception("Found %d fragments, must be 2." % nfrags)

        # Grab monomers
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name('monomerA')
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name('monomerB')

        # Compute monomer properties
        psi.set_active_molecule(monomerA)
        self.V_A = np.array(psi.MintsHelper().ao_potential())
        self.rhfA = energy('RHF')
        self.wfnA = psi.wavefunction()
        print("RHF for monomer A finished.")
        
        psi.set_active_molecule(monomerB)
        self.V_B = np.array(psi.MintsHelper().ao_potential())
        self.rhfB = energy('RHF')
        self.wfnB = psi.wavefunction()
        print("RHF for monomer B finished.")

        
        # Setup a few variables
        self.memory = memory

        # Save the wfn objects
        self.nmo = self.wfnA.nmo()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy() 
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        self.idx_A = ['a', 'r']

        self.C_A = np.array(self.wfnA.Ca())
        self.Co_A = self.C_A[:, :self.ndocc_A]
        self.Cv_A = self.C_A[:, self.ndocc_A:]
        self.D_A = np.einsum('ui,vi->uv', self.Co_A, self.Co_A)
        self.eps_A = np.array([self.wfnA.epsilon_a().get(x) for x in range(self.C_A.shape[0])])

        # Monomer B 
        self.nuc_rep_B = monomerB.nuclear_repulsion_energy() 
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B
        self.idx_B = ['b', 's']

        self.C_B = np.array(self.wfnB.Ca())
        self.Co_B = self.C_B[:, :self.ndocc_B]
        self.Cv_B = self.C_B[:, self.ndocc_B:]
        self.D_B = np.einsum('ui,vi->uv', self.Co_B, self.Co_B)
        self.eps_B = np.array([self.wfnB.epsilon_b().get(x) for x in range(self.C_B.shape[0])])

        # Dimer 
        self.nuc_rep = dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B 

        # Make some dicts
        self.slices =   {'a' : slice(0, self.ndocc_A),
                         'r' : slice(self.ndocc_A, None),
                         'b' : slice(0, self.ndocc_B),
                         's' : slice(self.ndocc_B, None)
                         }
        self.orbitals = {'a' : self.Co_A,     
                         'r' : self.Cv_A,     
                         'b' : self.Co_B,     
                         's' : self.Cv_B
                         }
        self.sizes =    {'a' : self.ndocc_A,     
                         'r' : self.nvirt_A,     
                         'b' : self.ndocc_B,     
                         's' : self.nvirt_B
                         }

        # Compute size of ERI tensor in GB
        psi.set_active_molecule(dimer)
        mints = psi.MintsHelper()
        ERI_Size = (self.nmo**4)*8.0 / 1E9
        print "Size of the ERI tensor will be %4.2f GB." % ERI_Size
        memory_footprint = ERI_Size*5
        if memory_footprint > self.memory:
            clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
        
        # Integral generation from Psi4's MintsHelper
        self.I = np.array(mints.ao_eri()).reshape(self.nmo, self.nmo, self.nmo, self.nmo).swapaxes(1,2)
        self.S = np.array(mints.ao_overlap()) 

        # Save additional rank 2 tensors
        self.V_A_BB = np.einsum('ui,vj,uv->ij', self.C_B, self.C_B, self.V_A)
        self.V_B_AA = np.einsum('ui,vj,uv->ij', self.C_A, self.C_A, self.V_B)
        self.V_A_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_A)
        self.V_B_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_B)

        self.S_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.S)


        print("\n...finished inializing SAPT object in %5.2f seconds.\n" % (time.time()-t))

    # Compute v on the fly
    def v(self, string):
        if len(string)!=4:
            print 'Compute V: string %s does not have 4 elements'
            exit()

        v = np.einsum('iP,ijkl->Pjkl', self.orbitals[string[0]], self.I)
        v = np.einsum('jQ,Pjkl->PQkl', self.orbitals[string[1]], v)
        v = np.einsum('kR,PQkl->PQRl', self.orbitals[string[2]], v)
        v = np.einsum('lS,PQRl->PQRS', self.orbitals[string[3]], v)
        return v

    # Grab S
    def s(self, string):
        if len(string)!=2:
            print 'Grab S: string %s does not have 2 elements'
            exit()

        s1 = string[0]
        s2 = string[1]

        # Compute on the fly
        # return np.einsum('ui,vj,uv->ij', self.orbitals[string[0]], self.orbitals[string[1]], self.S)

        # Same monomer and index- return diaganol
        if (s1==s2):
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

    # Grab eps:
    def eps(self, string, dim=1):
        if len(string)!=1:
            print 'Grab Epsilon: string %s does not have 1 elements'
            exit()

        shape = (-1,) + tuple([1]*(dim-1))

        if (string=='b') or (string=='s'):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            return self.eps_A[self.slices[string]].reshape(shape)

    # Grab MO potential
    def potential(self, string, side):
        monA = ['a','r']
        monB = ['b','s']

        s1 = string[0]
        s2 = string[1]

        # Two seperate cases
        if side=='A':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_A) / (2 * self.ndocc_A)

            if (s1 in self.idx_B) and (s2 in self.idx_B):
                return self.V_A_BB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_A)
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_A_AB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_A)
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_A_AB[self.slices[s2], self.slices[s1]].T / (2 * self.ndocc_A)
            else:
                print 'No match for %s indices in sapt.potential' % string

        if side=='B':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_B) / (2 * self.ndocc_B)

            if (s1 in self.idx_A) and (s2 in self.idx_A):
                return self.V_B_AA[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_B)
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_B_AB[self.slices[s1], self.slices[s2]] / (2 * self.ndocc_B)
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_B_AB[self.slices[s2], self.slices[s1]].T / (2 * self.ndocc_B)
            else:
                print 'No match for %s indices in sapt.potential' % string


    # Compute V tilde, Index as V_(1,2)^(3,4)
    # v0123 + V_a13 s02 / Na + V_b02 s13 / Nb
    def vt(self, string):
        if len(string)!=4:
            print 'Compute tilde{V}: string %s does not have 4 elements'
            exit()

        # ERI term
        V = self.v(string)

        # Potential A
        S_A = self.s(string[0] + string[2])
        V_A = self.potential(string[1] + string[3], 'A') 
        V += np.einsum('ik,jl->ijkl', S_A, V_A)

        # Potential B
        S_B = self.s(string[1] + string[3])
        V_B = self.potential(string[0] + string[2], 'B') 
        V += np.einsum('ik,jl->ijkl', V_B, S_B)

        # Nuclear- needs some help
        coef = self.nuc_rep / (4 * self.ndocc_A * self.ndocc_B)
        V += np.einsum('ik,jl->ijkl', S_A, S_B) * coef

        return V

# End SAPT helper
