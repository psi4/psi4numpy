"""
Computes the Linearized CCD AKA CEPA(0) without singles, correlation energy.
Equations taken by linearizing and density fitting Eq. 153 of [Crawford:2000:33].

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
R, F = integrals_DF(mol)
Fo = F["oo"].diagonal()
Fv = F["vv"].diagonal()
D = Fo.reshape(-1, 1, 1, 1) + Fo.reshape(-1, 1, 1) - Fv.reshape(-1, 1) - Fv
t2 = np.zeros(D.shape)
dsd = DirectSumDiis(3, 8)
nvir = len(Fv)

### Main Loop
for i in range(1, maxiter + 1):

    ### Two Electron Terms

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

    # r2 += g^AkIc t^JkBc -> IJAB
    temp = np.einsum("IAq, kcq, JkBc -> IJAB", R["ov"], R["ov"], t2, optimize=True)
    temp -= np.einsum("kIq, Acq, JkBc -> IJAB", R["oo"], R["vv"], t2, optimize=True)
    r2 += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3))

    ### One Electron Terms. For canonical orbitals, this will reduce to -t2, after dividing by D
    temp = -np.einsum("Ii, iJAB -> IJAB", F["oo"], t2, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = +np.einsum("aA, IJaB -> IJAB", F["vv"], t2, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))

    ### Step
    t2 += r2 / D
    t2 = dsd.diis(r2, t2)
    r_norm = np.linalg.norm(r2)
    Elccd = 0.5 * np.einsum("iaq, jbq, ijab", R["ov"], R["ov"], t2, optimize=True)
    print(f"{i:3d} E={Elccd:3.10f} R = {r_norm:0.8f}")
    if r_norm < float(f"1e-{target_convergence}"):
        break
else:
    raise Exception("Equations did not converge.")

if compare_psi4:
    wfn = psi4.energy("lccd", return_wfn=True)[1]

    # Change to wfn.variable when DFOCC variables are available on the wfn.
    psi4.compare_values(psi4.variable("CURRENT CORRELATION ENERGY"), Elccd, target_convergence, "LCCD Energy")
