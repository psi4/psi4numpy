"""
A simple Psi4 input script to compute EP3 using spin-orbitals.

References:  
- Equations for EP3 derived by the author from framework provided in [Szabo:1996], pp. 387-392.
- See "Further Reading" in [Szabo:1996], pp. 409, for additional details.
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
O -0.0247847074 0.0000000 -0.0175254347
H  0.0232702345 0.0000000  0.9433790708
H  0.8971830624 0.0000000 -0.2925203027
symmetry c1
""")

psi4.set_options({'basis': '6-31G',
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
ERI_Size = (nmo**4) * (2**4) * 8.0 / 1E9
print("\nSize of the SO ERI tensor will be %4.2f GB." % ERI_Size)
memory_footprint = ERI_Size * 2.2
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." %
                    (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
I = np.array(mints.ao_eri())
I = I.reshape(nmo, nmo, nmo, nmo)

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
t = time.time()
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

def get_slice(char, orb):
    """Returns either occupied, orbital, or virtual slice"""
    if char in 'abcdefgh':
        return slice(0, nocc)
    elif char in 'ij':
        return orb
    else:
        return slice(nocc, MO.shape[0])

def EP_term(n, orbital, factor, string, eps_views):
    """
    n = EPn order of theory
    orbital = orbital number
    factor = symmetry considerations
    string = summation string MO tensor than energy denominators
            - First n terms will be MO views (numerator)
            - Last n-1 terms will be epsilon view (denominator)
    eps_view = epsilon views
    Continously writing views seem like too much effort, why not automate the process?
    """

    #if not isinstance(eps_views, list):
    #    eps_view = list(eps_views)

    # Get slices
    slices = string.split(',')
    if len(slices) != (n * 2 - 1):
        clean()
        raise Exception('Number of terms does not match the order of pertubation theory')

    # Create views
    views = []

    # MO views
    for term in range(n):
        tmp_slice = slices[term]
        tmp_slice = [get_slice(x, orbital) for x in tmp_slice]
        views.append(MO[tmp_slice[0], tmp_slice[1], tmp_slice[2], tmp_slice[3]])

    # Add epsilon views
    views = views + eps_views

    # Remove i and j indices
    string = string.replace('i', '')
    string = string.replace('j', '')

    # Compute term!
    string += '->'
    term = factor * np.einsum(string, *views)
    return term

ep2_arr = []
ep3_arr = []
for orbital in range(nocc - num_orbs * 2, nocc, 2):
    E = eps[orbital]
    ep2_conv = False
    ep3_conv = False

    # EP2
    for ep_iter in range(50):
        Eold = E

        # Build energy denominators
        epsilon1 = 1 / (E + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)
        epsilon2 = 1 / (E + evirt.reshape(-1, 1, 1) - eocc.reshape(-1, 1) - eocc)
        epsilon1_2 = np.power(epsilon1, 2)
        epsilon2_2 = np.power(epsilon2, 2)

        # Compute sigma's
        sigma1 = EP_term(2, orbital, 0.5, 'iapq,pqja,apq', [epsilon1])
        sigma2 = EP_term(2, orbital, 0.5, 'ipab,abip,pab', [epsilon2])
        Enew = eps[orbital] + sigma1 + sigma2

        # Break if below threshold
        if abs(Enew - Eold) < 1.e-4:
            ep2_arr.append(Enew * 27.21138505)
            ep2_conv = True
            break

        # # Build derivatives
        sigma_deriv1 = EP_term(2, orbital, -0.5, 'iapq,pqja,apq', [epsilon1_2])
        sigma_deriv2 = EP_term(2, orbital, -0.5, 'ipab,abip,pab', [epsilon2_2])
        deriv = 1 - sigma_deriv1 - sigma_deriv2

        # Newton-Raphson update
        E = Eold - (E - Enew) / deriv


    if ep2_conv is False:
        ep2_arr.append(E)
        print('WARNING: EP2 for orbital HOMO - %d did not converged' % (ndocc - orbital/2 - 1))

    EP2 = E
    # EP3

    # EP3 self energy independant terms
    eps_ov = 1 / (eocc.reshape(-1, 1) - evirt)
    eps_oovv = 1 / (eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)
    eps_vvoo = 1 / (evirt.reshape(-1, 1, 1, 1) + evirt.reshape(-1, 1, 1) - eocc.reshape(-1, 1) - eocc)

    eps_oovv_2 = np.power(eps_oovv, 2)
    eps_vvoo_2 = np.power(eps_vvoo, 2)


    ep3_const  = EP_term(3, orbital,  0.5, 'ipja,qrpb,abqr,ap,abqr', [eps_ov, eps_oovv])
    ep3_const += EP_term(3, orbital, -0.5, 'ipjb,bqac,acpq,bp,acpq', [eps_ov, eps_oovv])

    ep3_const += EP_term(3, orbital,  0.5, 'prab,iqjp,abqr,abpr,abqr', [eps_oovv, eps_oovv])
    ep3_const += EP_term(3, orbital, -0.5, 'pqbc,ibja,acpq,bcpq,acpq', [eps_oovv, eps_oovv])

    ep3_const += EP_term(3, orbital,  0.5, 'prab,qbpr,iajq,abpr,aq', [eps_oovv, eps_ov])
    ep3_const += EP_term(3, orbital, -0.5, 'pqbc,bcaq,iajp,bcpq,ap', [eps_oovv, eps_ov])

    # Create KP-EP2 average geuss for EP3
    E = (EP2 + eps[orbital]) / 2
    for ep_iter in range(50):
        Eold = E

        # Compute energy denominators
        eps_eovv = 1 / (E + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)
        eps_evoo = 1 / (E + evirt.reshape(-1, 1, 1) - eocc.reshape(-1, 1) - eocc)
        eps_eovv_2 = np.power(eps_eovv, 2)
        eps_evoo_2 = np.power(eps_evoo, 2)


        Enew = eps[orbital] + ep3_const
        Ederiv = 0

        # EP2
        Enew += EP_term(2, orbital, 0.5, 'iapq,pqja,apq', [eps_eovv])
        Ederiv += EP_term(2, orbital, -0.5, 'iapq,pqja,apq', [eps_eovv_2])

        Enew += EP_term(2, orbital, 0.5, 'ipab,abip,pab', [eps_evoo])
        Ederiv += EP_term(2, orbital, -0.5, 'ipab,abip,pab', [eps_evoo_2])


        #EP3
        Enew += EP_term(3, orbital,  0.25, 'iaqs,qspr,prja,apr,aqs', [eps_eovv, eps_eovv])
        Ederiv += EP_term(3, orbital,  -0.25, 'iaqs,qspr,prja,apr,aqs', [eps_eovv_2, eps_eovv])
        Ederiv += EP_term(3, orbital,  -0.25, 'iaqs,qspr,prja,apr,aqs', [eps_eovv, eps_eovv_2])

        Enew += EP_term(3, orbital, -1.00, 'iaqr,qbpa,prjb,bpr,aqr', [eps_eovv, eps_eovv])
        Ederiv += EP_term(3, orbital,  1.00, 'iaqr,qbpa,prjb,bpr,aqr', [eps_eovv_2, eps_eovv])
        Ederiv += EP_term(3, orbital,  1.00, 'iaqr,qbpa,prjb,bpr,aqr', [eps_eovv, eps_eovv_2])

        # Block
        Enew += EP_term(3, orbital, -1.00, 'iraq,abpr,pqjb,bpq,abpr', [eps_eovv, eps_oovv])
        Enew += EP_term(3, orbital,  0.25, 'icab,abpq,pqjc,cpq,abpq', [eps_eovv, eps_oovv])
        Enew += EP_term(3, orbital, -1.00, 'ibpr,pqab,arjq,bpr,abpq', [eps_eovv, eps_oovv])
        Enew += EP_term(3, orbital,  0.25, 'ibpq,pqac,acjb,bpq,acpq', [eps_eovv, eps_oovv])

        Ederiv += EP_term(3, orbital,  1.00, 'iraq,abpr,pqjb,bpq,abpr', [eps_eovv_2, eps_oovv])
        Ederiv += EP_term(3, orbital, -0.25, 'icab,abpq,pqjc,cpq,abpq', [eps_eovv_2, eps_oovv])
        Ederiv += EP_term(3, orbital,  1.00, 'ibpr,pqab,arjq,bpr,abpq', [eps_eovv_2, eps_oovv])
        Ederiv += EP_term(3, orbital, -0.25, 'ibpq,pqac,acjb,bpq,acpq', [eps_eovv_2, eps_oovv])

        # Block
        Enew += EP_term(3, orbital, -0.25, 'iqab,abpr,prjq,qab,prab', [eps_evoo, eps_vvoo])
        Enew += EP_term(3, orbital,  1.00, 'iqac,abpq,pcjb,qac,pqab', [eps_evoo, eps_vvoo])
        Enew += EP_term(3, orbital, -0.25, 'irpq,pqab,abjr,rab,pqab', [eps_evoo, eps_vvoo])
        Enew += EP_term(3, orbital,  1.00, 'icpb,pqac,abjq,qab,pqac', [eps_evoo, eps_vvoo])

        Ederiv += EP_term(3, orbital,  0.25, 'iqab,abpr,prjq,qab,prab', [eps_evoo_2, eps_vvoo])
        Ederiv += EP_term(3, orbital, -1.00, 'iqac,abpq,pcjb,qac,pqab', [eps_evoo_2, eps_vvoo])
        Ederiv += EP_term(3, orbital,  0.25, 'irpq,pqab,abjr,rab,pqab', [eps_evoo_2, eps_vvoo])
        Ederiv += EP_term(3, orbital, -1.00, 'icpb,pqac,abjq,qab,pqac', [eps_evoo_2, eps_vvoo])

        # Block
        Enew += EP_term(3, orbital,  1.00, 'ipbc,bqap,acjq,pbc,qac', [eps_evoo, eps_evoo])
        Ederiv += EP_term(3, orbital, -1.00, 'ipbc,bqap,acjq,pbc,qac', [eps_evoo_2, eps_evoo])
        Ederiv += EP_term(3, orbital, -1.00, 'ipbc,bqap,acjq,pbc,qac', [eps_evoo, eps_evoo_2])

        Enew += EP_term(3, orbital, -0.25, 'ipbd,bdac,acjp,pbd,pac', [eps_evoo, eps_evoo])
        Ederiv += EP_term(3, orbital,  0.25, 'ipbd,bdac,acjp,pbd,pac', [eps_evoo_2, eps_evoo])
        Ederiv += EP_term(3, orbital,  0.25, 'ipbd,bdac,acjp,pbd,pac', [eps_evoo, eps_evoo_2])

        # Break if below threshold
        if abs(Enew - Eold) < 1.e-4:
            print('EP3 HOMO - %d converged in %d iterations' % ((ndocc - orbital/2 - 1), ep_iter))
            ep3_arr.append(Enew * 27.21138505)
            ep3_conv = True
            break

        # Newton-Raphson update
        E = Eold - (Eold - Enew) / (1 - Ederiv)

    if ep3_conv is False:
        ep3_arr.append(E)
        print('WARNING: EP3 for orbital HOMO - %d did not converged' % (ndocc - orbital/2 - 1))


print("\nKP - Koopmans' Theorem")
print("EP2 - Electron Propagator 2\n")
print("HOMO - n         KP (eV)              EP2 (eV)              EP3 (eV)")
print("---------------------------------------------------------------------")

KP_arr = eps[:nocc][::2] * 27.21138505

for orbital in range(0, len(ep2_arr)):
    print("% 4d     % 16.4f    % 16.4f    % 16.4f" % ((len(ep2_arr) - orbital - 1), KP_arr[orbital], ep2_arr[orbital],
                                                      ep3_arr[orbital]))

# 13.46 11.27
