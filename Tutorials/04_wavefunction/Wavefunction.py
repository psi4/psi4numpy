import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)

molecule mol {
O
H 1 1.1
H 1 1.1 2 105
symmetry c1
}

set {
 scf_type pk
 basis = sto-3g
}

# Compute the RHF energy
energy('RHF')

# Grab the wavefunction from the global space and save it.
wfn = wavefunction()
a = wfn.Ca()

print('Number of alpha electrons: %d' % wfn.nalpha())
print('Number of beta electrons: %d' % wfn.nbeta())
print('Number of double occupied orbitals: %d' % wfn.doccpi()[0])
print('The current energy %3.5f' %  wfn.energy())

print('\n The current orbitals:')
C = np.array(wfn.Ca())
print(C)

print('\n The density matrix:')
D = np.array(wfn.Da())
print(D)

print('\n The current fock matrix:')
F = np.array(wfn.Fa())
print(F)

print('\n The curreng eigenvalues:')
eps = wfn.epsilon_a()
eps = np.array([eps.get(x) for x in range(C.shape[0])])
print(eps)


# Compute the SCF energy using information from the wavefunction object
nbf = wfn.nmo()
Enuc = mol.nuclear_repulsion_energy()

mints = MintsHelper()
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())

SCF_E = np.sum((T + V + F) * D) + Enuc
print('\nThe compute SCF energy matches the energy of the wavefunction object: %s' % np.allclose(SCF_E, wfn.energy()))




