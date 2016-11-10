import numpy as np
import psi4

# Numpy print options
np.set_printoptions(precision=5, linewidth=200, suppress=True)

# Redirect output to a file
psi4.core.set_output_file("output.dat", False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 105
symmetry c1
""")


# Compute the SCF/sto-3g energy and return the wavefunction
rhf_e, wfn = psi4.energy('SCF/sto-3g', molecule=mol, return_wfn=True)

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
eps = np.array(wfn.epsilon_a())
print(eps)


# Compute the SCF energy using information from the wavefunction object
nbf = wfn.nmo()
Enuc = mol.nuclear_repulsion_energy()

mints = psi4.core.MintsHelper(wfn.basisset())
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())

SCF_E = np.sum((T + V + F) * D) + Enuc
print('\nThe compute SCF energy matches the energy of the wavefunction object: %s' % np.allclose(SCF_E, wfn.energy()))




