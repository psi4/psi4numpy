"""
A simple Psi4 input script to compute EP2 using spatial orbitals

References:
- EP2 energy expression from [Szabo:1996] page 391, Eqn. 7.39
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# Orbital to start at
start_orbs = 2

# Number of orbitals to compute
num_orbs = 4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

print('\nStarting RHF and integral build...')
t = time.time()

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from wavfunction class
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())

# Compute size of ERI tensor in GB
ERI_Size = (nmo**4) * 8e-9
print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

print('Building MO integrals.')
# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
C = wfn.Ca()
MO = np.asarray(mints.mo_eri(C, C, C, C))

# Grab ndocc shape
ndocc = np.asarray(wfn.Ca_subset("AO", "OCC")).shape[1]
nvirt = MO.shape[0] - ndocc

Eocc = eps[:ndocc]
Evir = eps[ndocc:]

# (pq|rs) -> <ps|rq>
MO = MO.swapaxes(1, 2)
print('Shape of MO integrals: %s' % str(MO.shape))
print('\n...finished RHF and integral build in %.3f seconds.\n' % (time.time() - t))


# Create occupied and virtual slices
o = slice(0, ndocc)
v = slice(ndocc, MO.shape[0])

if num_orbs > ndocc:
    num_orbs = ndocc

ep2_arr = []
for orbital in range(start_orbs + 1, start_orbs + num_orbs + 1):
    E = eps[orbital]
    ep2_conv = False

    for ep_iter in range(20):
        Eold = E

        # Build energy denominators
        epsilon1 = 1 / (E + Eocc.reshape(-1, 1, 1) - Evir.reshape(-1, 1) - Evir)
        epsilon2 = 1 / (E + Evir.reshape(-1, 1, 1) - Eocc.reshape(-1, 1) - Eocc)

        # Compute sigma's
        tmp1 = (2 * MO[orbital, o, v, v] - MO[o, orbital, v, v])
        sigma1 = np.einsum('rsa,ars,ars->', MO[v, v, orbital, o], tmp1, epsilon1)

        tmp2 = (2 * MO[orbital, v, o, o] - MO[v, orbital, o, o])
        sigma2 = np.einsum('abr,rab,rab->', MO[o, o, orbital, v], tmp2, epsilon2)
        Enew = eps[orbital] + sigma1 + sigma2

        # Break if below threshold
        if abs(Enew - Eold) < 1.e-4:
            ep2_conv = True
            ep2_arr.append(Enew * 27.21138505)
            break

        # Build derivatives
        sigma_deriv1 = np.einsum('rsa,ars,ars->', MO[v, v, orbital, o], tmp1, np.power(epsilon1, 2))
        sigma_deriv2 = np.einsum('abr,rab,rab->', MO[o, o, orbital, v], tmp2, np.power(epsilon2, 2))
        deriv = -1 * (sigma_deriv1 + sigma_deriv2)

        # Newton-Raphson update
        E = Eold - (Eold - Enew) / (1 - deriv)

    if ep2_conv is False:
        ep2_arr.append(E * 27.21138505)
        print('WARNING: EP2 for orbital HOMO - %d did not converged' % (ndocc - orbital - 1))

print("KP - Koopmans' Theorem")
print("EP2 - Electron Propagator 2\n")
print("HOMO - n         KP (eV)              EP2 (eV)")
print("----------------------------------------------")

KP_arr = eps * 27.21138505

for orbital in range(0, len(ep2_arr)):
    kp_orb = start_orbs + orbital + 1
    print("% 4d     % 16.4f    % 16.4f" % (kp_orb - ndocc + 1, KP_arr[kp_orb], ep2_arr[orbital]))



