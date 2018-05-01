"""
A simple Psi4 input script to compute EP2 using spin-orbitals.

References:
- EP2 SO energy expression from [Szabo:1996] pp. 390, Eqn. 7.38
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

# Number of orbitals below the HOMO to compute
num_orbs = 5

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set Psi4 Options
psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Coefficient Matrix
C = np.array(wfn.Ca())
# Double occupied orbitals
ndocc = wfn.doccpi()[0]
# Number of molecular orbitals
nmo = wfn.nmo()
# SCF energy
SCF_E = wfn.energy()
# Orbital energies
eps = wfn.epsilon_a()
eps = np.array([eps.get(x) for x in range(C.shape[0])])


# Compute size of SO-ERI tensor in GB
ERI_Size = (nmo**4)*(2**4)*8.0 / 1E9
print("\nSize of the SO ERI tensor will be %4.2f GB." % ERI_Size)
memory_footprint = ERI_Size*2.2
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
I = np.array(mints.ao_eri())
I = I.reshape(nmo, nmo, nmo, nmo)

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time()-t))


#Make spin-orbital MO
t=time.time()
print('Starting AO -> spin-orbital MO transformation...')
nso = nmo * 2

MO = np.einsum('rJ,pqrs->pqJs', C, I)
MO = np.einsum('pI,pqJs->IqJs', C, MO)
MO = np.einsum('sB,IqJs->IqJB', C, MO)
MO = np.einsum('qA,IqJB->IAJB', C, MO)

# Tile MO array so that we have alternating alpha/beta spin orbitals
MO = np.repeat(MO, 2, axis=0)
MO = np.repeat(MO, 2, axis=1)
MO = np.repeat(MO, 2, axis=2)
MO = np.repeat(MO, 2, axis=3)

# Build spin mask
spin_ind = np.arange(nso, dtype=np.int) % 2
spin_mask = (spin_ind.reshape(-1, 1, 1, 1) == spin_ind.reshape(-1, 1, 1))
spin_mask = spin_mask * (spin_ind.reshape(-1, 1) == spin_ind)

# compute antisymmetrized MO integrals
MO *= spin_mask
MO = MO - MO.swapaxes(1, 3)
MO = MO.swapaxes(1, 2)
print('..finished transformation in %.3f seconds.\n' % (time.time()-t))

# Update nocc and nvirt
nocc = ndocc * 2
nvirt = MO.shape[0] - nocc

# Build epsilon tensor
eps = np.repeat(eps, 2)
eocc = eps[:nocc]
evirt = eps[nocc:]

# Create occupied and virtual slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

if num_orbs > nocc:
    num_orbs = nocc

ep2_arr = []
for orbital in range(nocc-num_orbs*2, nocc, 2):
    E = eps[orbital]
    ep2_conv = False

    for ep_iter in range(50):
        Eold = E

        # Build energy denominators
        epsilon1 = 1/(E + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)
        epsilon2 = 1/(E + evirt.reshape(-1, 1, 1) - eocc.reshape(-1, 1) - eocc)

        # Compute sigma's
        sigma1 = 0.5 * np.einsum('rsa,ars,ars', MO[v, v, orbital, o], MO[orbital, o, v, v], epsilon1)
        sigma2 = 0.5 * np.einsum('abr,rab,rab', MO[o, o, orbital, v], MO[orbital, v, o, o], epsilon2)
        Enew = eps[orbital] + sigma1 + sigma2

        # Break if below threshold
        if abs(Enew - Eold) < 1.e-4:
            ep2_conv = True
            ep2_arr.append(Enew * 27.21138505)
            break

        # Build derivatives
        sigma_deriv1 = -1 * np.einsum('rsa,ars,ars', MO[v, v, orbital, o], MO[orbital, o, v, v], np.power(epsilon1, 2))
        sigma_deriv2 = -1 * np.einsum('abr,rab,rab', MO[o, o, orbital, v], MO[orbital, v, o, o], np.power(epsilon2, 2))
        deriv = 1 - (sigma_deriv1 + sigma_deriv2)

        # Newton-Raphson update
        E = Eold - (Eold - Enew) / deriv

    if ep2_conv is False:
        ep2_arr.append(Enew * 27.21138505)
        print('WARNING: EP2 for orbital HOMO - %d did not converged' % (ndocc - orbital/2 - 1))


print("KP - Koopmans' Theorem")
print("EP2 - Electron Propagator 2\n")
print("HOMO - n         KP (eV)              EP2 (eV)")
print("----------------------------------------------")

KP_arr = eps[:nocc][::2] * 27.21138505

for orbital in range(0, len(ep2_arr)):
    print("% 4d     % 16.4f    % 16.4f" % ((len(ep2_arr)-orbital-1), KP_arr[orbital], ep2_arr[orbital]))



