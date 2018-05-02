"""
Reference implementation of the MP3 correlation energy utilizing antisymmetrized
spin-orbitals from an RHF reference.

References:
- Equations from [Szabo:1996]
"""

__authors__    = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from 
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())


# Compute size of ERI tensor in GB
ERI_Size = (nmo ** 4) * 8e-9
print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

#Make spin-orbital MO
t=time.time()
print('Starting ERI build and spin AO -> spin-orbital MO transformation...')
mints = psi4.core.MintsHelper(wfn.basisset())
MO = np.asarray(mints.mo_spin_eri(C, C))
eps = np.repeat(eps, 2)
nso = nmo * 2

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

# Update nocc and nvirt
nocc = ndocc * 2
nvirt = MO.shape[0] - nocc

# Build epsilon tensor
eocc = eps[:nocc]
evir = eps[nocc:]
epsilon = 1/(eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evir.reshape(-1, 1) - evir)

# Create occupied and virtual slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

# MP2 Correlation: [Szabo:1996] pp. 352, Eqn 6.72
MP2corr_E = 0.25 * np.einsum('abrs,rsab,abrs', MO[o, o, v, v], MO[v, v, o, o], epsilon)
MP2total_E = SCF_E + MP2corr_E
print('MP2 correlation energy:      %16.10f' % MP2corr_E)
print('MP2 total energy:            %16.10f' % MP2total_E)

# Compare to Psi4
psi4.compare_values(psi4.energy('MP2'), MP2total_E, 6, 'MP2 Energy')

# MP3 Correlation: [Szabo:1996] pp. 353, Eqn. 6.75
eqn1 = 0.125 * np.einsum('abrs,cdab,rscd,abrs,cdrs->', MO[o, o, v, v], MO[o, o, o, o], MO[v, v, o, o], epsilon, epsilon)
eqn2 = 0.125 * np.einsum('abrs,rstu,tuab,abrs,abtu', MO[o, o, v, v], MO[v, v, v, v], MO[v, v, o, o], epsilon, epsilon)
eqn3 = np.einsum('abrs,cstb,rtac,absr,acrt', MO[o, o, v, v], MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)

MP3corr_E = eqn1 + eqn2 + eqn3
MP3total_E = MP2total_E + MP3corr_E
print('\nMP3 correlation energy:      %16.10f' % MP3corr_E)
print('MP3 total energy:            %16.10f' % MP3total_E)

# Compare to Psi4
psi4.compare_values(psi4.energy('MP3'), MP3total_E, 6, 'MP3 Energy')



