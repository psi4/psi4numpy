"""
A Psi4 input script to compute CI energy using an iterative Davidson-Lu solver.

References:
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import numpy as np
import psi4

np.set_printoptions(precision=7, linewidth=200, threshold=2000, suppress=True)

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

## Uncomment for short test
#mol = psi4.geometry("""
#0 1
#N 0.0 0.0 0.0
#N 1.1 0.0 0.0
#symmetry c1
#units angstrom
#""")
## Number of roots
#nroot = 1
#psi4.set_options({"BASIS": "STO-3G", "NUM_ROOTS" : 1})

## Uncomment for long test
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
 # Number of roots
nroot = 1
psi4.set_options({"BASIS": "6-31g", "NUM_ROOTS" : 3})

# Build the SCF Wavefunction
scf_energy, scf_wfn = psi4.energy("HF", return_wfn=True)

# Build integrals
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Build a CI Wavefunction
# This automatically generates the determinants based on the options
# Note that a CISD wavefunction is default if no options are given
# Other CI wavefunctions can be requested, e.g. { "FCI" : True }
psi4.core.prepare_options_for_module("DETCI")
ciwfn = psi4.core.CIWavefunction(scf_wfn)

# Transform the integrals
mints.integrals()
ciwfn.transform_ci_integrals()

# Get the number of determinants
ndet = ciwfn.ndet()
print("Number of determinants in CI space:  %d" % ciwfn.ndet())


## Other options

# Number of guess vectors
guess_size = 4

# Convergence tolerance of the residual norm
ctol = 1.e-5

# Convergence tolerance of the energy
etol = 1.e-9

# Make sure the guess is smaller than the CI space
if guess_size > ndet:
    raise Exception("Number of guesses (%d)  exceeds CI dimension (%d)!" % (guess_size, ndet))

print('Using %d determinants in the guess\n' % guess_size)

# Build the Hamiltonian in the space of guess determinants
H = np.array(ciwfn.hamiltonian(guess_size))

# Get guess eigenvectors
gvecs = []
gevals, gevecs = np.linalg.eigh(H)
#for x in range(nroot):
for x in range(guess_size):
    guess = np.zeros((ciwfn.ndet()))
    guess[:guess_size] = gevecs[:, x]
    gvecs.append(guess)
    print('Guess CI energy (Hsize %d)   %2.9f' % (guess_size, gevals[x] + mol.nuclear_repulsion_energy()))
print("")

# Maximum number of vectors
max_guess = 200

# Build diagonal
Hd = ciwfn.Hd_vector(5)

cvecs = ciwfn.new_civector(max_guess, 200, True, True)
cvecs.set_nvec(max_guess)
cvecs.init_io_files(False)

swork_vec = max_guess
svecs = ciwfn.new_civector(max_guess + 1, 201, True, True)
svecs.set_nvec(max_guess)
svecs.init_io_files(False)

dwork_vec = nroot
dvecs = ciwfn.new_civector(nroot + 1, 202, True, True)
dvecs.init_io_files(False)
dvecs.set_nvec(nroot + 1)

for x in range(nroot + 1):
    dvecs.write(x, 0)
for x in range(max_guess):
    svecs.write(x, 0)
for x in range(max_guess):
    cvecs.write(x, 0)

# Current number of vectors
num_vecs = guess_size

# Copy gvec data into in ci_gvecs
arr_cvecs = np.asarray(cvecs)
for x in range(guess_size):
    arr_cvecs[:] = gvecs[x]
    cvecs.write(x, 0)
    cvecs.symnormalize(1 / np.linalg.norm(gvecs[x]), x)

delta_c = np.zeros(nroot)

Eold = scf_energy
G = np.zeros((max_guess, max_guess))

# Begin Davidson iterations
for CI_ITER in range(max_guess - 1):

    # Subspace Matrix, Gij = < bi | H | bj >
    for i in range(0, num_vecs):
        # Build sigma for each b
        cvecs.read(i, 0)
        svecs.read(i, 0)
        ciwfn.sigma(cvecs, svecs, i, i)
        for j in range(i, num_vecs):
            # G_ij = (b_i, sigma_j)
            cvecs.read(i, 0)
            svecs.read(j, 0)
            G[j, i] = G[i, j] = svecs.vdot(cvecs, i, j)

    evals, evecs = np.linalg.eigh(G[:num_vecs, :num_vecs])
    CI_E = evals

    # Use average over roots as convergence criteria
    avg_energy = 0.0
    avg_dc = 0.0
    for n in range(nroot):
        avg_energy += evals[n]
        avg_dc += delta_c[n]
    avg_energy /= nroot
    avg_dc /= nroot
    avg_energy += mol.nuclear_repulsion_energy()

    print('CI Iteration %3d: Energy = %4.16f   dE = % 1.5E   dC = %1.5E' % (CI_ITER, avg_energy, (avg_energy - Eold),
                                                                            avg_dc))
    if (abs(avg_energy - Eold) < etol) and (avg_dc < ctol) and (CI_ITER > 3):
        print('CI has converged!\n')
        break
    Eold = avg_energy

    # Build new vectors as linear combinations of the subspace matrix, H
    for n in range(nroot):

        # Build as linear combinations of previous vectors
        dvecs.zero()
        dvecs.write(dwork_vec, 0)
        for c in range(len(evecs[:, n])):
            dvecs.axpy(evecs[c, n], cvecs, dwork_vec, c)

        # Build new vector new_vec = ((H * cvec) - evals[n] * cvec) / (evals[n] - Hd)
        ciwfn.sigma(dvecs, svecs, dwork_vec, swork_vec)
        svecs.axpy(-1 * evals[n], dvecs, swork_vec, dwork_vec)
        norm = svecs.dcalc(evals[n], Hd, swork_vec)

        if (norm < 1e-9):
            continue

        svecs.symnormalize(1 / norm, swork_vec)
        delta_c[n] = norm

        # Build a new vector that is orthornormal to all previous vectors
        dvecs.copy(svecs, n, swork_vec)
        norm = dvecs.norm(n)
        dvecs.symnormalize(1 / norm, n)

        total_proj = 0
        for i in range(num_vecs):
            proj = svecs.vdot(cvecs, swork_vec, i)
            total_proj += proj
            dvecs.axpy(-proj, cvecs, n, i)

        norm = dvecs.norm(n)
        dvecs.symnormalize(1 / norm, n)

        # This *should* screen out contributions that are projected out by above
        if True:
            cvecs.write(num_vecs, 0)
            cvecs.copy(dvecs, num_vecs, n)
            num_vecs += 1

print('SCF energy:           % 16.10f' % (scf_energy))
for n in range(nroot):
    print('State %d Total Energy: % 16.10f' % (n, CI_E[n] + mol.nuclear_repulsion_energy()))
print("")

E = psi4.energy('detci')

for n in range(nroot):
    ci_ref = psi4.get_variable('CI ROOT %d TOTAL ENERGY' % n)
    ci_compute = CI_E[n] + mol.nuclear_repulsion_energy()
    psi4.compare_values(ci_ref, ci_compute, 6, 'CI Root %d Total Energy' % n)
