"""
A reference implementation to compute the first dipole
hyperpolarizability $\beta$ from a restricted HF reference using the
$2n+1$ rule from perturbation theory.

References:
Equations taken from [Karna:1991:487], http://dx.doi.org/10.1002/jcc.540120409
"""

__authors__   =  "Eric J. Berquist"
__credits__   = ["Eric J. Berquist"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-08-26"

from itertools import permutations, product

import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_CPHF import helper_CPHF

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file("output.dat", False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set options for CPHF
psi4.set_options({"basis": "aug-cc-pvdz",
                  "scf_type": "direct",
                  "df_scf_guess": False,
                  "e_convergence": 1e-9,
                  "d_convergence": 1e-9})

# Compute the (first) hyperpolarizability corresponding to static
# fields, beta(0;0,0), eqns. (IV-2a) and (VII-4).

helper = helper_CPHF(mol)
# For the $2n+1$ rule, the quadratic response starting quantities must
# come from linear response.
helper.run()

na = np.newaxis
moenergies = helper.epsilon
C = np.asarray(helper.C)
Co = helper.Co
Cv = helper.Cv
nbf, norb = C.shape
nocc = Co.shape[1]
nvir = norb - nocc
nov = nocc * nvir
x = np.asarray(helper.x)
ncomp = x.shape[0]
integrals_ao = np.asarray([np.asarray(dipole_ao_component)
                           for dipole_ao_component in helper.tmp_dipoles])

# form full MO-basis dipole integrals
integrals_mo = np.empty(shape=(ncomp, norb, norb))
for i in range(ncomp):
    integrals_mo[i] = (C.T).dot(integrals_ao[i]).dot(C)

# repack response vectors to [norb, norb]; 1/2 is due to X + Y
U = np.zeros_like(integrals_mo)
for i in range(ncomp):
    U[i, :nocc, nocc:] = 0.5 * x[i].reshape(nocc, nvir)
    U[i, nocc:, :nocc] = -0.5 * x[i].reshape(nocc, nvir).T

# form G matrices from perturbation and generalized Fock matrices; do
# one more Fock build for each response vector
jk = psi4.core.JK.build(helper.scf_wfn.basisset())
jk.initialize()
G = np.empty_like(U)
R = psi4.core.Matrix(nbf, nocc)
npR = np.asarray(R)
for i in range(ncomp):
    V = integrals_mo[i]

    # eqn. (III-1b) Note: this simplified handling of the response
    # vector transformation for the Fock build is insufficient for
    # frequency-dependent response.
    jk.C_clear()
    # Psi4's JK builders don't take a density, but a left set of
    # coefficients with shape [nbf, nocc] and a right set of
    # coefficents with shape [nbf, nocc]. Because the response vector
    # describes occ -> vir transitions, we perform ([nocc, nvir] *
    # [nbf, nvir]^T)^T.
    L = Co
    npR[:] = x[i].reshape(nocc, nvir).dot(np.asarray(Cv).T).T
    jk.C_left_add(L)
    jk.C_right_add(R)
    jk.compute()
    # 1/2 is due to X + Y
    J = 0.5 * np.asarray(jk.J()[0])
    K = 0.5 * np.asarray(jk.K()[0])

    # eqn. (21b)
    F = (C.T).dot(4 * J - K.T - K).dot(C)
    G[i] = V + F

# form epsilon matrices, eqn. (34)
E = G.copy()
omega = 0
for i in range(ncomp):
    eoU = (moenergies[..., na] + omega) * U[i]
    Ue = U[i] * moenergies[na]
    E[i] += (eoU - Ue)

# Assume some symmetry and calculate only part of the tensor.
# eqn. (VII-4)
hyperpolarizability = np.zeros(shape=(6, 3))
off1 = [0, 1, 2, 0, 0, 1]
off2 = [0, 1, 2, 1, 2, 2]
for r in range(6):
    b = off1[r]
    c = off2[r]
    for a in range(3):
        tl1 = 2 * np.trace(U[a].dot(G[b]).dot(U[c])[:nocc, :nocc])
        tl2 = 2 * np.trace(U[a].dot(G[c]).dot(U[b])[:nocc, :nocc])
        tl3 = 2 * np.trace(U[c].dot(G[a]).dot(U[b])[:nocc, :nocc])
        tr1 = np.trace(U[c].dot(U[b]).dot(E[a])[:nocc, :nocc])
        tr2 = np.trace(U[b].dot(U[c]).dot(E[a])[:nocc, :nocc])
        tr3 = np.trace(U[c].dot(U[a]).dot(E[b])[:nocc, :nocc])
        tr4 = np.trace(U[a].dot(U[c]).dot(E[b])[:nocc, :nocc])
        tr5 = np.trace(U[b].dot(U[a]).dot(E[c])[:nocc, :nocc])
        tr6 = np.trace(U[a].dot(U[b]).dot(E[c])[:nocc, :nocc])
        tl = tl1 + tl2 + tl3
        tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
        hyperpolarizability[r, a] = -2 * (tl - tr)

ref_static = np.array([
    [ 0.00000001,   0.00000000,   0.22843772],
    [ 0.00000000,   0.00000000, -25.35476040],
    [ 0.00000000,   0.00000000, -10.84023375],
    [ 0.00000000,   0.00000000,   0.00000000],
    [ 0.22843772,   0.00000000,   0.00000000],
    [ 0.00000000, -25.35476040,   0.00000000]
])
assert np.allclose(ref_static, hyperpolarizability, rtol=0.0, atol=1.0e-3)
print('\nFirst dipole hyperpolarizability (static):')
print(hyperpolarizability)

# Compute the (first) hyperpolarizability corresponding to
# second-harmonic generation, beta(-2w;w,w), eqns. (IV-2c) and
# (VII-1). Because two different frequencies are involved, the linear
# response equations must be solved twice.

print('Setting up for second-harmonic generation (SHG) calculation...')
# In SHG, the first frequency is doubled to obtain the second
# frequency. All variables containing '1' correspond to the first
# (set) frequency, and all variables containing '2' correspond to the
# second (doubled) frequency.
f1 = 0.0773178
f2 = 2 * f1

print('\nForming response vectors for {} a.u.'.format(f1))
helper1 = helper_CPHF(mol)
helper1.solve_dynamic_direct(omega=f1)
helper1.form_polarizability()
print(helper1.polar)
print('\nForming response vectors for {} a.u.'.format(f2))
helper2 = helper_CPHF(mol)
helper2.solve_dynamic_direct(omega=f2)
helper2.form_polarizability()
print(helper2.polar)

rspvecs1 = helper1.x
rspvecs2 = helper2.x

# repack response vectors to [norb, norb]
U1 = np.zeros_like(integrals_mo)
U2 = np.zeros_like(integrals_mo)
for i in range(ncomp):
    U1[i, :nocc, nocc:] = rspvecs1[i][nov:].reshape(nocc, nvir)
    U1[i, nocc:, :nocc] = rspvecs1[i][:nov].reshape(nocc, nvir).T
    U2[i, :nocc, nocc:] = rspvecs2[i][nov:].reshape(nocc, nvir)
    U2[i, nocc:, :nocc] = rspvecs2[i][:nov].reshape(nocc, nvir).T

G1 = np.empty_like(U1)
G2 = np.empty_like(U2)
R1_l = psi4.core.Matrix(nbf, nocc)
R1_r = psi4.core.Matrix(nbf, nocc)
R2_l = psi4.core.Matrix(nbf, nocc)
R2_r = psi4.core.Matrix(nbf, nocc)
npR1_l = np.asarray(R1_l)
npR1_r = np.asarray(R1_r)
npR2_l = np.asarray(R2_l)
npR2_r = np.asarray(R2_r)
jk.C_clear()
jk.C_left_add(Co)
jk.C_right_add(R1_l)
jk.C_left_add(Co)
jk.C_right_add(R1_r)
jk.C_left_add(Co)
jk.C_right_add(R2_l)
jk.C_left_add(Co)
jk.C_right_add(R2_r)
nCo = np.asarray(Co)
# Do 4 Fock builds at a time: X/Y vectors for both frequencies; loop
# over operator components
for i in range(3):
    V = integrals_mo[i]

    x1 = U1[i, :nocc, :]
    y1 = U1[i, :, :nocc]
    x2 = U2[i, :nocc, :]
    y2 = U2[i, :, :nocc]
    npR1_l[:] = C.dot(x1.T)
    npR1_r[:] = C.dot(y1)
    npR2_l[:] = C.dot(x2.T)
    npR2_r[:] = C.dot(y2)

    jk.compute()

    J1_l = -np.asarray(jk.J()[0])
    K1_l = -np.asarray(jk.K()[0])
    J1_r = np.asarray(jk.J()[1])
    K1_r = np.asarray(jk.K()[1])
    J2_l = -np.asarray(jk.J()[2])
    K2_l = -np.asarray(jk.K()[2])
    J2_r = np.asarray(jk.J()[3])
    K2_r = np.asarray(jk.K()[3])
    J1 = J1_l + J1_r
    J2 = J2_l + J2_r
    K1 = K1_l + K1_r.T
    K2 = K2_l + K2_r.T

    F1 = (C.T).dot(2 * J1 - K1).dot(C)
    F2 = (C.T).dot(2 * J2 - K2).dot(C)
    G1[i, ...] = V + F1
    G2[i, ...] = V + F2

# form epsilon matrices, eqn. (34), one for each frequency
E1 = G1.copy()
E2 = G2.copy()
for i in range(ncomp):
    eoU1 = (moenergies[..., na] + f1) * U1[i]
    Ue1 = U1[i] * moenergies[na]
    E1[i] += (eoU1 - Ue1)
    eoU2 = (moenergies[..., na] + f2) * U2[i]
    Ue2 = U2[i] * moenergies[na]
    E2[i] += (eoU2 - Ue2)

# Assume some symmetry and calculate only part of the tensor.

hyperpolarizability = np.zeros(shape=(6, 3))
for r in range(6):
    b = off1[r]
    c = off2[r]
    for a in range(3):
        tl1 = np.trace(U2[a].T.dot(G1[b]).dot(U1[c])[:nocc, :nocc])
        tl2 = np.trace(U1[c].dot(G1[b]).dot(U2[a].T)[:nocc, :nocc])
        tl3 = np.trace(U2[a].T.dot(G1[c]).dot(U1[b])[:nocc, :nocc])
        tl4 = np.trace(U1[b].dot(G1[c]).dot(U2[a].T)[:nocc, :nocc])
        tl5 = np.trace(U1[c].dot(-G2[a].T).dot(U1[b])[:nocc, :nocc])
        tl6 = np.trace(U1[b].dot(-G2[a].T).dot(U1[c])[:nocc, :nocc])
        tr1 = np.trace(U1[c].dot(U1[b]).dot(-E2[a].T)[:nocc, :nocc])
        tr2 = np.trace(U1[b].dot(U1[c]).dot(-E2[a].T)[:nocc, :nocc])
        tr3 = np.trace(U1[c].dot(U2[a].T).dot(E1[b])[:nocc, :nocc])
        tr4 = np.trace(U2[a].T.dot(U1[c]).dot(E1[b])[:nocc, :nocc])
        tr5 = np.trace(U1[b].dot(U2[a].T).dot(E1[c])[:nocc, :nocc])
        tr6 = np.trace(U2[a].T.dot(U1[b]).dot(E1[c])[:nocc, :nocc])
        tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
        tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
        hyperpolarizability[r, a] = 2 * (tl - tr)

# pylint: disable=C0326
ref = np.array([
    [ 0.00000000,   0.00000000,   1.92505358],
    [ 0.00000000,   0.00000000, -31.33652886],
    [ 0.00000000,   0.00000000, -13.92830863],
    [ 0.00000000,   0.00000000,   0.00000000],
    [-1.80626084,   0.00000000,   0.00000000],
    [ 0.00000000, -31.13504192,   0.00000000]
])
ref_avgs = np.array([0.00000000, 0.00000000, 45.69300223])
ref_avg = 45.69300223

thresh = 1.0e-2
# assert np.all(np.abs(ref - hyperpolarizability) < thresh)

print('hyperpolarizability: SHG, (-{}; {}, {}), symmetry-unique components'.format(f2, f1, f1))
print(hyperpolarizability)
print('ref')
print(ref)

# Transpose all frequency-doubled quantities (+2w) to get -2w.

for i in range(ncomp):
    U2[i] = U2[i].T
    G2[i] = -G2[i].T
    E2[i] = -E2[i].T

# Assume some symmetry and calculate only part of the tensor. This
# time, work with the in-place manipulated quantities (this tests
# their correctness).

mU = (U2, U1)
mG = (G2, G1)
me = (E2, E1)

hyperpolarizability = np.zeros(shape=(6, 3))
off1 = [0, 1, 2, 0, 0, 1]
off2 = [0, 1, 2, 1, 2, 2]
for r in range(6):
    b = off1[r]
    c = off2[r]
    for a in range(3):
        tl1 = np.trace(mU[0][a].dot(mG[1][b]).dot(mU[1][c])[:nocc, :nocc])
        tl2 = np.trace(mU[1][c].dot(mG[1][b]).dot(mU[0][a])[:nocc, :nocc])
        tl3 = np.trace(mU[0][a].dot(mG[1][c]).dot(mU[1][b])[:nocc, :nocc])
        tl4 = np.trace(mU[1][b].dot(mG[1][c]).dot(mU[0][a])[:nocc, :nocc])
        tl5 = np.trace(mU[1][c].dot(mG[0][a]).dot(mU[1][b])[:nocc, :nocc])
        tl6 = np.trace(mU[1][b].dot(mG[0][a]).dot(mU[1][c])[:nocc, :nocc])
        tr1 = np.trace(mU[1][c].dot(mU[1][b]).dot(me[0][a])[:nocc, :nocc])
        tr2 = np.trace(mU[1][b].dot(mU[1][c]).dot(me[0][a])[:nocc, :nocc])
        tr3 = np.trace(mU[1][c].dot(mU[0][a]).dot(me[1][b])[:nocc, :nocc])
        tr4 = np.trace(mU[0][a].dot(mU[1][c]).dot(me[1][b])[:nocc, :nocc])
        tr5 = np.trace(mU[1][b].dot(mU[0][a]).dot(me[1][c])[:nocc, :nocc])
        tr6 = np.trace(mU[0][a].dot(mU[1][b]).dot(me[1][c])[:nocc, :nocc])
        tl = [tl1, tl2, tl3, tl4, tl5, tl6]
        tr = [tr1, tr2, tr3, tr4, tr5, tr6]
        hyperpolarizability[r, a] = 2 * (sum(tl) - sum(tr))

assert np.all(np.abs(ref - hyperpolarizability) < thresh)

# Assume no symmetry and calculate the full tensor.

hyperpolarizability_full = np.zeros(shape=(3, 3, 3))

# components x, y, z
for ip, p in enumerate(list(product(range(3), range(3), range(3)))):
    a, b, c = p
    tl, tr = [], []
    # 1st tuple -> index a, b, c (*not* x, y, z!)
    # 2nd tuple -> index frequency (0 -> -2w, 1 -> +w)
    for iq, q in enumerate(list(permutations(zip(p, (0, 1, 1)), 3))):
        d, e, f = q
        tlp = (mU[d[1]][d[0]]).dot(mG[e[1]][e[0]]).dot(mU[f[1]][f[0]])
        tle = np.trace(tlp[:nocc, :nocc])
        tl.append(tle)
        trp = (mU[d[1]][d[0]]).dot(mU[e[1]][e[0]]).dot(me[f[1]][f[0]])
        tre = np.trace(trp[:nocc, :nocc])
        tr.append(tre)
    hyperpolarizability_full[a, b, c] = 2 * (sum(tl) - sum(tr))
print('hyperpolarizability: SHG, (-{}; {}, {}), full tensor'.format(f2, f1, f1))
print(hyperpolarizability_full)

# Check that the elements of the reduced and full tensors are
# equivalent.

for r in range(6):
    b = off1[r]
    c = off2[r]
    for a in range(3):
        diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
        assert abs(diff) < 1.0e-14
