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
#Be 0.0 0.0 0.0
#symmetry c1
#""")
#psi4.set_options({"BASIS": "6-31G",
#                 "ACTIVE" : [5] })

## Uncomment for long test
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")
psi4.set_options({"BASIS": "6-31g"})
                  # "FCI" : True })

# Build the SCF Wavefunction
scf_energy, scf_wfn = psi4.energy("HF", return_wfn=True)

# Build integrals
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Build a CI Wavefunction
# This automatically generates the determinants based on the options
# Note that a CISD wavefunction is default if no options are given
psi4.core.prepare_options_for_module("DETCI")
ciwfn = psi4.core.CIWavefunction(scf_wfn)

# Transform the integrals
mints.integrals()
ciwfn.transform_ci_integrals()


ndet = ciwfn.ndet()
print("Number of determinants in FCI space:  %d" % ciwfn.ndet())

# Number of guess vectors
guess_size = 2

if guess_size > ndet:
    raise Exception( "Number of guesses (%d)  exceeds FCI dimension (%d)!" % (guess_size, ndet))

print('Using %d determinants in the guess' % guess_size)
H = np.array(ciwfn.hamiltonian(guess_size))

num_eig = 1
ctol = 1.e-5
etol = 1.e-8

gvecs = []
gevals, gevecs = np.linalg.eigh(H)
#for x in range(num_eig):
for x in range(guess_size):
    guess = np.zeros((ciwfn.ndet()))
    guess[:guess_size] = gevecs[:, x]
    gvecs.append(guess)
    print( 'Guess CI energy (Hsize %d)   %2.9f' % (guess_size, gevals[x]))

# Build a few CI vectors
max_guess = 16

Hd = ciwfn.Hd_vector(5)

cvecs = ciwfn.new_civector(max_guess, 200, True, True)
cvecs.set_nvec(max_guess)
cvecs.init_io_files(False)

swork_vec = max_guess
svecs = ciwfn.new_civector(max_guess + 1, 201, True, True)
svecs.set_nvec(max_guess)
svecs.init_io_files(False)

dwork_vec = num_eig
dvecs = ciwfn.new_civector(num_eig + 1, 202, True, True)
dvecs.init_io_files(False)
dvecs.set_nvec(num_eig + 1)

for x in range(num_eig + 1):
    dvecs.write(x, 0)

# Current number of vectors
#num_vecs = num_eig
num_vecs = guess_size

# Copy gvec data into in ci_gvecs
arr_cvecs = np.asarray(cvecs)
for x in range(guess_size):
    arr_cvecs[:] = gvecs[x]
    cvecs.write(x, 0)

delta_c = 0.0

Eold = scf_energy
G = np.zeros((max_guess, max_guess))

for CI_ITER in range(max_guess - 1):

    # Subspace Matrix, Gij = < bi | H | bj >
    for i in range(0, num_vecs):
        # Build sigma for each b
        ciwfn.sigma(cvecs, svecs, i, i)
        for j in range(i, num_vecs):
            # G_ij = (b_i, sigma_j)
            G[j,i] = G[i, j] = svecs.vdot(cvecs, i, j)

    evals, evecs = np.linalg.eigh(G[:num_vecs, :num_vecs])
    CI_E = evals[0]
    print('CI Iteration %3d: Energy = %4.16f   dE = % 1.5E   dC = %1.5E'
          % (CI_ITER, CI_E, (CI_E - Eold), delta_c))
    if (abs(CI_E - Eold) < etol) and (delta_c < ctol) and (CI_ITER > 3):
        print('CI has converged!')
        break
    Eold = CI_E

    # Build new vectors as linear combinations of the subspace matrix, H
    for n in range(num_eig):

        # Build as linear combinations of previous vectors
        dvecs.zero()
        dvecs.write(dwork_vec, 0)
        for c in range(len(evecs[:,n])):
            dvecs.axpy(evecs[c, n], cvecs, dwork_vec, c)

        # Build new vector new_vec = ((H * cvec) - evals[n] * cvec) / (evals[n] - Hd)
        ciwfn.sigma(dvecs, svecs, dwork_vec, swork_vec)
        svecs.axpy(-1 * evals[n], dvecs, swork_vec, dwork_vec)
        norm = svecs.dcalc(evals[n], Hd, swork_vec)
        svecs.symnormalize(1 / norm, swork_vec)
        delta_c = norm

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


CI_E = CI_E + mol.nuclear_repulsion_energy()

print('SCF energy:         % 16.10f' % (scf_energy))
print('FCI correlation:    % 16.10f' % (CI_E - scf_energy))
print('Total FCI energy:   % 16.10f' % (CI_E))

psi4.driver.p4util.compare_values(psi4.energy('CISD'), CI_E, 6, 'FCI Energy')
