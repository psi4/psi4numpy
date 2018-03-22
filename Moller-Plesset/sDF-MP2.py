"""
A reference implementation of stochastic orbital resolution of identity
(or density-fitted) MP2 (sRI-MP2) from a RHF reference.

Reference: 
Stochastic Formulation of the Resolution of Identity: Application to
Second Order Moller-Plesset Perturbation Theory
J. Chem. Theory Comput., 2017, 13 (10), pp 4605-4610
DOI: 10.1021/acs.jctc.7b00343

Tyler Y. Takeshita 
Department of Chemistry, University of California Berkeley
Materials Sciences Division, Lawrence Berkeley National Laboratory

Wibe A. de Jong
Computational Research Division, Lawrence Berkeley National Laboratory

Daniel Neuhauser
Department of Chemistry and Biochemistry, University of California, Los Angeles

Roi Baer
Fritz Harber Center for Molecular Dynamics, Institute of Chemistry, The Hebrew University of Jerusalem

Eran Rabani
Department of Chemistry, University of California Berkeley
Materials Sciences Division, Lawrence Berkeley National Laboratory
The Sackler Center for Computational Molecular Science, Tel Aviv University
"""

__authors__   = ["Tyler Y. Takeshita", "Daniel G. A. Smith"]
__credits__   = ["Tyler Y. Takeshita", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-04-14"

import numpy as np
import psi4
import time

# Set numpy defaults
np.set_printoptions(precision=5, linewidth=200, suppress=True)
# psi4.set_output_file("output.dat")

# ==> Geometry <==
# Note: Symmetry was turned off
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104
symmetry c1
""")

# How many samples to run?
nsample = 5000

# ==> Basis sets <==
psi4.set_options({
    'basis': 'aug-cc-pvdz',
    'scf_type': 'df',
    'e_convergence': 1e-10,
    'd_convergence': 1e-10
})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))

# Build auxiliary basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "RIFIT", "aug-cc-pVDZ") 

# Get orbital basis & build zero basis
orb = wfn.basisset()

# The zero basis set
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)

# ==> Build Density-Fitted Integrals <==

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)

# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)

# Remove excess dimensions of Ppq, & metric
Ppq = np.squeeze(Ppq)
metric = np.squeeze(metric)

# Build the Qso object
Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)

# ==> Perform HF <==
energy, scf_wfn = psi4.energy('scf', return_wfn=True)
print("Finished SCF...")

# ==> AO -> MO Transformation of integrals <==
# Get the MO coefficients and energies
evecs = np.array(scf_wfn.epsilon_a())

Co = scf_wfn.Ca_subset("AO", "OCC")
Cv = scf_wfn.Ca_subset("AO", "VIR")

Qia = np.einsum("Qpq,pi,qa->Qia", Qpq, Co, Cv, optimize=True)

nocc = scf_wfn.nalpha()
nvirt = scf_wfn.nmo() - nocc

denom = 1.0 / (evecs[:nocc].reshape(-1, 1, 1, 1) + evecs[:nocc].reshape(-1, 1, 1) -
               evecs[nocc:].reshape(-1, 1) - evecs[nocc:])

t = time.time()
e_srimp2 = 0.0
print("Transformed ERIs...")

# Loop over samples to reduce stochastic noise
print("Starting sample loop...")
for x in range(nsample):

    # ==> Build Stochastic Integral Matrices <==
    # Create two random vector
    vec = np.random.choice([-1, 1], size=(Qia.shape[0]))
    vecp = np.random.choice([-1, 1], size=(Qia.shape[0]))

    # Generate first R matrices
    ia = np.einsum("Q,Qia->ia", vec, Qia)
    iap = np.einsum("Q,Qia->ia", vecp, Qia)

    # ==> Calculate a single stochastic RI-MP2 (sRI-MP2) sample <==

    # Caculate sRI-MP2 correlation energy
    e_srimp2 += 2.0 * np.einsum('ijab,ia,ia,jb,jb->', denom, ia, iap, ia, iap)
    e_srimp2 -= np.einsum('ijab,ia,ib,jb,ja->', denom, ia, iap, ia, iap)

e_srimp2 /= float(nsample)
total_time = time.time() - t
time_per_sample = total_time / float(nsample)

# Print sample energy to output
print("\nNumber of samples:                 % 16d" % nsample)
print("Total time (s):                    % 16.2f" % total_time)
print("Time per sample (us):              % 16.2f" % (time_per_sample * 1.e6))
print("sRI-MP2 correlation sample energy: % 16.10f" % e_srimp2)

psi_mp2_energy = psi4.energy("MP2")
mp2_correlation_energy = psi4.get_variable("MP2 CORRELATION ENERGY")

print("\nRI-MP2 energy:                     % 16.10f" % mp2_correlation_energy)
print("Sample error                       % 16.10f" % (e_srimp2 - mp2_correlation_energy))
