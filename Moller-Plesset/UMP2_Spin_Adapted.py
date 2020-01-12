"""
A reference implementation of MP2 from a UHF reference in the spatial orbital basis.
Intended to show basic example of spin adapting a spin orbital code.
"""

__authors__ = "Matthew McAllister Davis"
__credits__ = ["Matthew McAllister Davis", "Justin M. Turney"]
__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2020-01-08"

import psi4
import numpy as np
from time import time

np.set_printoptions(precision=10, linewidth=200, suppress=True)
psi4.core.be_quiet()
psi4.core.set_output_file("output.dat")
psi4.set_memory('2 GB')

# Computing on ethane cation!
mol = psi4.geometry("""
        1 2
        O
        H 1 1.1
        H 1 1.1 2 104.0
        symmetry c1
        """)

compare = True
psi4.set_options({
    'basis': 'sto-3g',
    'reference': 'uhf',
    'mp2_type': 'conv',
    'e_convergence': 1e-12,
    'd_convergence': 1e-10,
    'scf_type': 'pk'
})

# Get SCF wavefunction and basic info
print('Starting RHF.')
t = time()
e, wfn = psi4.energy('scf', return_wfn=True, mol=mol)
t = time() - t
print(f'Computed @RHF {e:16.10f} Eh in {t:4.6f} s.\n')
nmo = wfn.nmo()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

# size of occ and vir spaces for alpha and beta spin
occa = nalpha
vira = nmo - nalpha
occb = nbeta
virb = nmo - nbeta

# Core hamiltonian in AO basis
h = wfn.H().np

# Orbital eigenvalues for MP2 denominators
epsa = wfn.epsilon_a().np
epsb = wfn.epsilon_b().np
epsa_occ = epsa[:occa]
epsa_vir = epsa[occa:]
epsb_occ = epsb[:occb]
epsb_vir = epsb[occb:]

# Make a MintsHelper for integral transformation
mints = psi4.core.MintsHelper(wfn.basisset())

# occ and vir slices for alpha and beta
oa = slice(0, nalpha)
va = slice(nalpha, nmo)
ob = slice(0, nbeta)
vb = slice(nbeta, nmo)

# AO->MO coefficients, occ and vir subsets
Coa = wfn.Ca_subset("AO", "OCC")
Cob = wfn.Cb_subset("AO", "OCC")
Cva = wfn.Ca_subset("AO", "VIR")
Cvb = wfn.Cb_subset("AO", "VIR")

# Build MO basis TEI
t = time()
print('Transforming TEI from AO -> MO.')
mo_ijab = mints.mo_eri(Coa, Cva, Coa, Cva).np
mo_iJaB = mints.mo_eri(Coa, Cva, Cob, Cvb).np
mo_IJAB = mints.mo_eri(Cob, Cvb, Cob, Cvb).np
t = time() - t
print(f'Transformed TEI in {t:4.6f} s.\n')

# Form antisymmetrized MO integrals
# 0 1 2 3
# i j a b
# (0,2,1,3) -> chemists notation to physicists notation
# (0,2,3,1) -> chemists notation to physicists notation and interchange a,b
t = time()
print('Antisymmetrizing TEI.')
ijab = mo_ijab.transpose(0, 2, 1, 3) - mo_ijab.transpose(0, 2, 3, 1)
iJaB = mo_iJaB.transpose(0, 2, 1, 3)
IJAB = mo_IJAB.transpose(0, 2, 1, 3) - mo_IJAB.transpose(0, 2, 3, 1)
t = time() - t
print(f'Antisymmetrized TEI in {t:4.6f} s.\n')

t = time()
print('Building denominator arrays and T2 arrays.')

# Compute denominator arrays
Dijab = epsa_occ.reshape(-1, 1, 1, 1) + epsa_occ.reshape(
    -1, 1, 1) - epsa_vir.reshape(-1, 1) - epsa_vir
tijab = ijab / Dijab
DiJaB = epsa_occ.reshape(-1, 1, 1, 1) + epsb_occ.reshape(
    -1, 1, 1) - epsa_vir.reshape(-1, 1) - epsb_vir
tiJaB = iJaB / DiJaB
DIJAB = epsb_occ.reshape(-1, 1, 1, 1) + epsb_occ.reshape(
    -1, 1, 1) - epsb_vir.reshape(-1, 1) - epsb_vir
tIJAB = IJAB / DIJAB
t = time() - t
print(f'Denominator and T2 arrays finished in {t:4.6f} s.\n')

# Compute the MP2 energy
t = time()
print('Computing the UMP2 energy.')
emp2 = (1 / 4) * np.einsum('ijab,ijab->', tijab, ijab)
emp2 += np.einsum('ijab,ijab->', tiJaB, iJaB)
emp2 += (1 / 4) * np.einsum('ijab,ijab->', tIJAB, IJAB)
t = time() - t
print(f"Computed MP2 correction {emp2:16.10f} Eh in {t:4.6f} s.")
if compare:
    emp2_psi = psi4.energy('mp2', mol=mol)
    print("Does the psi4numpy energy match Psi4? ",
          np.allclose(emp2_psi - e, emp2))
