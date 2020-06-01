"""
Computes the Linearized CCSD AKA CEPA(0) with singles, correlation energy.
Equations taken by linearizing Eq. 152 and 153 of [Crawford:2000:33].

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import numpy as np
import psi4
from DSD import DirectSumDiis
from integrals import integrals

### Settings

mol = psi4.geometry("""
O
H 1 1.0
H 1 1.0 2 104.5
symmetry c1""")
scf_type = "pk"
target_convergence = 7
freeze_core = True
maxiter = 50
compare_psi4 = True
basis = "cc-pvdz"

### Setup
psi4.set_options({"freeze_core": freeze_core, "scf_type": scf_type, "e_convergence": target_convergence + 1, "basis": basis})
I, F = integrals(mol, singles=True)
t1 = np.zeros(F["ov"].shape)
t2 = np.zeros(I["oovv"].shape)
Fo = F["oo"].diagonal()
Fv = F["vv"].diagonal()
D1 = Fo.reshape(-1, 1) - Fv
D2 = Fo.reshape(-1, 1, 1, 1) + Fo.reshape(-1, 1, 1) - Fv.reshape(-1, 1) - Fv
dsd = DirectSumDiis(3, 8)

### Main Loop
for i in range(1, maxiter + 1):
    ### R1
    ## Two Electron Terms
    r1 = 0.5 * np.einsum("iAab, iIab -> IA", I["ovvv"], t2, optimize=True)
    r1 -= 0.5 * np.einsum("jkIa, jkAa -> IA", I["ooov"], t2, optimize=True)
    r1 += np.einsum("aI iA, ia -> IA", I["voov"], t1, optimize=True)
    ## One Electron Terms
    r1 += 0.5 * np.einsum("ia, IiAa -> IA", F["ov"], t2, optimize=True)
    r1 += F["ov"] # This term is zero by Brillouin's Theorem for UHF or closed-shell RHF references
    # For canonical orbtials, these next terms will reduce to -t1, after dividing by D1
    r1 += np.einsum("aA, Ia -> IA", F["vv"], t1, optimize=True)
    r1 -= np.einsum("Ii, iA -> IA", F["oo"], t1, optimize=True)

    ### R2
    # Two Electron Terms
    r2 = I["oovv"] + 0.5 * np.einsum("ABcd, IJcd -> IJAB", I["vvvv"], t2, optimize=True)
    r2 += 0.5 * np.einsum("klIJ, klAB -> IJAB", I["oooo"], t2, optimize=True)
    temp = np.einsum("AkIc, JkBc -> IJAB", I["voov"], t2, optimize=True)
    r2 += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3))
    # One Electron Terms. For canonical orbitals, this will reduce to -t2, after dividing by D2
    temp = - np.einsum("Ii, iJAB -> IJAB", F["oo"], t2, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = + np.einsum("aA, IJaB -> IJAB", F["vv"], t2, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))
    # Singles Terms: New compared to LCCD
    temp = + np.einsum("IaAB, Ja -> IJAB", I["ovvv"], t1, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = - np.einsum("IJiB, iA -> IJAB", I["ooov"], t1, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))

    ### Step
    t1 += r1 / D1
    t2 += r2 / D2
    t2, t1 = dsd.diis([r2, r1], [t2, t1])
    r_norm = np.sqrt(np.linalg.norm(r2) ** 2 + np.linalg.norm(r1) ** 2)
    # We linearize the singles in the energy function as well. See eq. 12 of [Taube:2009:144112].
    # For canonical orbitals, F_ia vanishes, so some quantum chemistry codes (Psi4's `fnocc`) neglect it.
    Elccsd = 0.25 * np.sum(I["oovv"] * t2) + np.sum(F["ov"] * t1) 
    print(f"{i:3d} E={Elccsd:3.10f} R = {r_norm:0.8f}")
    if r_norm < float(f"1e-{target_convergence}"):
        break
else:
    raise Exception("Equations did not converge.")

if compare_psi4:
   wfn = psi4.energy("lccsd", return_wfn=True)[1]
   psi4.compare_values(psi4.variable("CURRENT CORRELATION ENERGY"), Elccsd, target_convergence, "LCCSD Energy")
