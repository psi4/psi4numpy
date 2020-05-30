"""
Computes the Orbital-Optimized LCCD correlation energy.
Equations taken from [Bozkaya:2013:054104].

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import numpy as np
import psi4
from scipy import linalg as spla
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
maxiter = 50
compare_psi4 = True
basis = "cc-pvdz"

### Setup
psi4.set_options({"scf_type": scf_type, "e_convergence": target_convergence + 1, "basis": basis})
I, F, intermed = integrals(mol, singles=True, return_intermediates=True)
t1 = np.zeros(F["ov"].shape)
t2 = np.zeros(I["oovv"].shape)
dsd = DirectSumDiis(3, 8)
num_occ, num_vir = t1.shape

# We need the initial one-electron integrals as well for orbital-optimized methods.
# The orbital gradient expression requires these instead of the Fock integrals that our amplitude expressions need.
H = {
    "oo": np.einsum('pP, qQ, pq -> PQ', intermed["O"], intermed["O"], intermed["OEI"], optimize = True),
    "ov": np.einsum('pP, qQ, pq -> PQ', intermed["O"], intermed["V"], intermed["OEI"], optimize = True),
    "vv": np.einsum('pP, qQ, pq -> PQ', intermed["V"], intermed["V"], intermed["OEI"], optimize = True)
    }
Escf = mol.nuclear_repulsion_energy() + np.trace(H["oo"]) + 0.5 * np.einsum("ijij ->", I["oooo"], optimize = True)

### Main Loop
for i in range(1, maxiter + 1):

    if i != 1:
        # Compute new orbitals and transform into them. See Section IIA5 of Bozkaya.
        Zoo = np.zeros((num_occ, num_occ))
        Zvv = np.zeros((num_vir, num_vir))
        X = np.block([[Zoo, -t1],
                     [t1.T, Zvv]])
        U = spla.expm(X)
        C = np.hstack((intermed["O"], intermed["V"]))
        C = C @ U
        C_O, C_V = np.hsplit(C, [num_occ])
        I = {
            "oovv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_V, C_V, intermed["TEI"], optimize = True),
            "oooo": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_O, intermed["TEI"], optimize = True),
            "voov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_O, C_O, C_V, intermed["TEI"], optimize = True),
            "vvvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_V, C_V, C_V, intermed["TEI"], optimize = True),
            "ovvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_V, C_V, C_V, intermed["TEI"], optimize = True),
            "ooov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_V, intermed["TEI"], optimize = True)
            }
        H = {
            "oo": np.einsum('pP, qQ, pq -> PQ', C_O, C_O, intermed["OEI"], optimize = True),
            "ov": np.einsum('pP, qQ, pq -> PQ', C_O, C_V, intermed["OEI"], optimize = True),
            "vv": np.einsum('pP, qQ, pq -> PQ', C_V, C_V, intermed["OEI"], optimize = True)
            }
        F = {
            "oo": H["oo"] + np.einsum('iP iQ -> PQ', I["oooo"], optimize = True),
            "ov": H["ov"] + np.einsum('iP iQ -> PQ', I["ooov"], optimize = True),
            "vv": H["vv"] - np.einsum('Pi iQ -> PQ', I["voov"], optimize = True)
            }

    Fo = F["oo"].diagonal()
    Fv = F["vv"].diagonal()
    D1 = Fo.reshape(-1, 1) - Fv
    D2 = Fo.reshape(-1, 1, 1, 1) + Fo.reshape(-1, 1, 1) - Fv.reshape(-1, 1) - Fv

    ### Construct reduced density matrices. Eq. 24 is always relevant.
    scf_opdm = np.eye(*F["oo"].shape) # Eq. 25
    opdm_oo_corr = - 0.5 * np.einsum("Ik ab,Jk ab->IJ", t2, t2, optimize=True) # Eq. 29
    opdm_oo = scf_opdm + opdm_oo_corr # OO block of eq. 23
    opdm_vv = + 0.5 * np.einsum("ij Ac,ij Bc->AB", t2, t2, optimize=True) # Eq. 30
    tpdm_oovv = t2.copy() # Eq. 34
    tpdm_vvvv = 0.5 * np.einsum("ijCD, ijAB -> ABCD", t2, t2, optimize=True) # Eq. 32
    tpdm_oooo = 0.5 * np.einsum("IJab, KLab -> IJKL", t2, t2, optimize=True) # Eq. 31
    # Eq. 24
    temp = np.einsum("p r, q s -> pqrs", scf_opdm, opdm_oo_corr, optimize=True)
    tpdm_oooo += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((1, 0, 2, 3)) - temp.transpose((0, 1, 3, 2))
    # Eq. 26
    temp = np.einsum("pr, qs -> pqrs", scf_opdm, scf_opdm, optimize=True)
    tpdm_oooo += temp - temp.transpose((0, 1, 3, 2))
    # Eq. 33
    tpdm_ovov = - np.einsum("IiBa, JiAa -> IAJB", t2, t2, optimize=True) + np.einsum("p r, q s -> pqrs", scf_opdm, opdm_vv, optimize=True)

    # Expand out eq. 37. We use Fock matrix elements for convenience. 
    r1 = np.einsum("iA, Ii -> IA", H["ov"], opdm_oo)
    r1 -= np.einsum("Ia, aA -> IA", H["ov"], opdm_vv)
    r1 -= 0.5 * np.einsum("jkiA, Iijk -> IA", I["ooov"], tpdm_oooo)
    r1 -= 0.5 * np.einsum("iAab, Iiab -> IA", I["ovvv"], tpdm_oovv)
    r1 += np.einsum("ibAa, Iaib -> IA", I["ovvv"], tpdm_ovov)
    r1 -= 0.5 * np.einsum("Iabc, Aabc -> IA", I["ovvv"], tpdm_vvvv)
    r1 -= 0.5 * np.einsum("ijIa, ijAa -> IA", I["ooov"], tpdm_oovv)
    r1 -= np.einsum("iIja, iAja -> IA", I["ooov"], tpdm_ovov)
    t1 += r1 / D1

    ### Construct T2 residual and step; Eq. 5
    # Two Electron Terms
    r2 = I["oovv"] + 0.5 * np.einsum("ABcd, IJcd -> IJAB", I["vvvv"], t2, optimize=True)
    r2 += 0.5 * np.einsum("klIJ, klAB -> IJAB", I["oooo"], t2, optimize=True)
    temp = np.einsum("AkIc, JkBc -> IJAB", I["voov"], t2, optimize=True)
    r2 += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3))
    # One Electron Terms. For canonical orbitals, this will reduce to -t2, after dividing by D
    temp = - np.einsum("Ii, iJAB -> IJAB", F["oo"], t2, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = + np.einsum("aA, IJaB -> IJAB", F["vv"], t2, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))
    # Step
    t2 += r2 / D2

    ### DIIS and Print
    t2, t1 = dsd.diis([r2, r1], [t2, t1])
    r_norm = np.sqrt(np.linalg.norm(r2) ** 2 + np.linalg.norm(r1) ** 2)
    Eref = mol.nuclear_repulsion_energy() + np.trace(H["oo"]) + 0.5 * np.einsum("ijij ->", I["oooo"], optimize = True)
    Eolccd = 0.25 * np.sum(I["oovv"] * t2) + Eref - Escf
    print(f"{i:3d} E={Eolccd:3.10f} R = {r_norm:0.8f}")
    if r_norm < float(f"1e-{target_convergence}"):
        break
else:
    raise Exception("Equations did not converge.")

if compare_psi4:
   wfn = psi4.energy("olccd", return_wfn=True)[1]
   psi4.compare_values(wfn.variable("CURRENT CORRELATION ENERGY"), Eolccd, target_convergence, "OLCCD Energy")
