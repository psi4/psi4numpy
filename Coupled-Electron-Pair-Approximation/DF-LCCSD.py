"""
Computes the Linearized CCSD AKA CEPA(0) with singles, correlation energy.
Equations taken by linearizing and density fitting Eq. 152 and 153 of [Crawford:2000:33].

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import numpy as np
import psi4
from DSD import DirectSumDiis
from integrals import integrals_DF

### Settings

mol = psi4.geometry("""
O
H 1 1.0
H 1 1.0 2 104.5
symmetry c1""")
scf_type = "df"
target_convergence = 7
freeze_core = True
maxiter = 50
compare_psi4 = True
basis = "cc-pvdz"

### Setup
psi4.set_options({
    "freeze_core": freeze_core,
    "scf_type": scf_type,
    "e_convergence": target_convergence + 1,
    "basis": basis,
    "cc_type": "df"
})
R, F = integrals_DF(mol, singles=True)
Fo = F["oo"].diagonal()
Fv = F["vv"].diagonal()
D1 = Fo.reshape(-1, 1) - Fv
D2 = Fo.reshape(-1, 1, 1, 1) + Fo.reshape(-1, 1, 1) - Fv.reshape(-1, 1) - Fv
t1 = np.zeros(D1.shape)
t2 = np.zeros(D2.shape)
dsd = DirectSumDiis(3, 8)
nvir = len(Fv)

### Main Loop
for i in range(1, maxiter + 1):

    ### R1: Two Electron Terms
    # r1 = 0.5 g^iA_ab t^iI_ab
    r1 = np.einsum("iaq, Abq, iIab -> IA", R["ov"], R["vv"], t2, optimize=True)

    # r1 -= 0.5 * g^jk_Ia t^jk_Aa
    r1 -= np.einsum("jIq, kaq, jkAa -> IA", R["oo"], R["ov"], t2, optimize=True)

    # r1 = g^aI_iA t^i_a
    r1 += np.einsum("iaq, IAq, ia -> IA", R["ov"], R["ov"], t1, optimize=True)
    r1 -= np.einsum("aAq, Iiq, ia -> IA", R["vv"], R["oo"], t1, optimize=True)

    ### R1: One Electron Terms
    r1 += 0.5 * np.einsum("ia, IiAa -> IA", F["ov"], t2, optimize=True)
    r1 += F["ov"] # This term is zero by Brillouin's Theorem for UHF or closed-shell RHF references
    # For canonical orbtials, these next terms will reduce to -t1, after dividing by D1
    r1 += np.einsum("aA, Ia -> IA", F["vv"], t1, optimize=True)
    r1 -= np.einsum("Ii, iA -> IA", F["oo"], t1, optimize=True)

    ### R2: Two Electron Terms

    # r2 = g^ij_ab
    temp = np.einsum("iaq, jbq -> ijab", R["ov"], R["ov"], optimize=True)
    r2 = temp - temp.transpose((1, 0, 2, 3))

    # r2 += 0.5 g^AB_cd t^IJ_cd -> IJAB
    # Trying to do this computation by direct einsum leads to reassembling the full VVVV block of integrals.
    # We would rather not store a V^4 intermediate in memory if we can avoid it.
    # Accordingly, we loop over a V index.
    for A in range(nvir):
        temp = np.einsum("cq, Bdq, IJcd -> IJB", R["vv"][A], R["vv"], t2, optimize=True)
        r2[:, :, A, :] += temp

    # r2 += 0.5 g^kl_IJ t^kl_AB
    r2 += np.einsum("kIq, lJq, klAB -> IJAB", R["oo"], R["oo"], t2, optimize=True)

    # r2 += P(IJ/AB) g^AkIc t^JkBc -> IJAB
    temp = np.einsum("IAq, kcq, JkBc -> IJAB", R["ov"], R["ov"], t2, optimize=True)
    temp -= np.einsum("kIq, Acq, JkBc -> IJAB", R["oo"], R["vv"], t2, optimize=True)
    r2 += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3))

    ### R2: One Electron Terms. For canonical orbitals, this will reduce to -t2, after dividing by D
    temp = -np.einsum("Ii, iJAB -> IJAB", F["oo"], t2, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = +np.einsum("aA, IJaB -> IJAB", F["vv"], t2, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))

    ### R2: Singles Terms. These are new compared to LCCD.
    # r2 += P(IJ) g^IaAB t^J_a
    temp = + np.einsum("IAq, aBq, Ja -> IJAB", R["ov"], R["vv"], t1, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3)) - temp.transpose((0, 1, 3, 2)) + temp.transpose((1, 0, 3, 2))

    # r2 += P(AB) g^IJ_iB t^i_A
    temp = - np.einsum("Iiq, JBq, iA -> IJAB", R["oo"], R["ov"], t1, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3)) + temp.transpose((1, 0, 3, 2))

    ### Step
    t1 += r1 / D1
    t2 += r2 / D2
    t2, t1 = dsd.diis([r2, r1], [t2, t1])
    r_norm = np.sqrt(np.linalg.norm(r2) ** 2 + np.linalg.norm(r1) ** 2)
    Elccsd = 0.5 * np.einsum("iaq, jbq, ijab", R["ov"], R["ov"], t2, optimize=True) + np.sum(F["ov"] * t1)
    print(f"{i:3d} E={Elccsd:3.10f} R = {r_norm:0.8f}")
    if r_norm < float(f"1e-{target_convergence}"):
        break
else:
    raise Exception("Equations did not converge.")

if compare_psi4:
    # Psi doesn't currently have DF-LCCSD implemented, so no test against Psi is possible.
    psi4.compare_values(-0.217981606, Elccsd, target_convergence, "LCCSD Energy")
