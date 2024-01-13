"""
A reference implementation of Gaussian-n composite methods.
"""

__authors__ = "Eric J. Berquist"
__credits__ = ["Eric J. Berquist"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-09-30"

import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

print("\n Performing a G1 calculation.\n")

# The starting geometry is not critical, since all compound methods include
# one or more geometry optiimizations.
mol = psi4.geometry("""
O          0.000000000000     0.000000000000    -0.075791843589
H          0.000000000000    -0.866811828967     0.601435779270
H          0.000000000000     0.866811828967     0.601435779270
symmetry c1
""")

psi4.set_options({
    "basis": "6-31G*",
    "scf_type": "direct",
    "df_scf_guess": False,
    "g_convergence": "gau",
})

print("1a. Geometry optimization: HF/6-31G*")
psi4.optimize("HF")
print("1b. Harmonic frequencies: HF/6-31G*")
psi4.frequencies("HF")

T = psi4.core.get_global_option("T")
zpe = psi4.core.scalar_variable("ZPVE")
# du = psi4.core.scalar_variable("INTERNAL ENERGY CORRECTION")
du = psi4.core.scalar_variable("THERMAL ENERGY CORRECTION")
dh = psi4.core.scalar_variable("ENTHALPY CORRECTION")
dg = psi4.core.scalar_variable("GIBBS FREE ENERGY CORRECTION")

psi4.core.clean()

print("2. Geometry optimization: MP2/6-31G* [no frozen core]")

psi4.set_options({
    "basis": "6-31G*",
    "scf_type": "direct",
    "df_scf_guess": False,
    "mp2_type": "conv",
    "freeze_core": "false",
    "g_convergence": "gau",
})

psi4.optimize("MP2")
psi4.core.clean()

print("3. Single-point energy: MP4(SDTQ)/6-311G** [frozen core]")

psi4.set_options({
    "basis": "6-311G**",
    "scf_type": "direct",
    "df_scf_guess": False,
    "freeze_core": "true",
})

e_step3, wfn = psi4.energy("MP4", return_wfn=True)
psi4.core.clean()

print("4. Single-point energy: MP4(SDTQ)/6-311+G** [frozen core]")

psi4.set_options({
    "basis": "6-311+G**",
    "scf_type": "direct",
    "df_scf_guess": False,
    "freeze_core": "true",
})

e_step4 = psi4.energy("MP4")
psi4.core.clean()

e_step4 -= e_step3

print("5. Single point energy: MP4(SDTQ)/6-311G**(2df) [frozen core]")

psi4.set_options({
    "basis": "6-311G(2df,p)",
    "scf_type": "direct",
    "df_scf_guess": False,
    "freeze_core": "true",
})

e_step5 = psi4.energy("MP4")
psi4.core.clean()

e_step5 -= e_step3

print("6. Single-point energy: QCISD(T)/6-311G** [frozen core]")

psi4.set_options({
    "basis": "6-311G**",
    "scf_type": "direct",
    "df_scf_guess": False,
    "freeze_core": "true",
})

e_step6 = psi4.energy("QCISD(T)")
psi4.core.clean()

e_step6 -= e_step3

e_combined = e_step3 + e_step4 + e_step5 + e_step6

print("7. High-level correction (HLC)")

nfc = wfn.nfrzc()
nalpha = wfn.nalpha() - nfc
nbeta = wfn.nbeta() - nfc
e_hlc = (-0.19 * nalpha) - (5.95 * nbeta)
e_hlc /= 1000

e_e = e_combined + e_hlc

print("8. ZPE addition")

scale_fac = 0.8929
e_zpe = scale_fac * zpe
e_thermal = du - (1 - scale_fac) * zpe
g1_0k = e_e + e_zpe
g1_energy = g1_0k - zpe + du
g1_enthalpy = g1_0k - zpe + dh
g1_free_energy = g1_0k - zpe + dg

print("E(ZPE)= {:10.6f}".format(e_zpe))
print("E(Thermal)= {:10.6f}".format(e_thermal))
print("E(QCISD(T))= {:10.6f}".format(e_step6 + e_step3))
print("E(Empiric)= {:10.6f}".format(e_hlc))
print("DE(Plus)= {:10.6f}".format(e_step4))
print("DE(2DF)= {:10.6f}".format(e_step5))
print("G1(0 K)= {:10.6f}".format(g1_0k))
print("G1 Energy= {:10.6f}".format(g1_energy))
print("G1 Enthalpy= {:10.6f}".format(g1_enthalpy))
print("G1 Free Energy= {:10.6f}".format(g1_free_energy))

print("\nAdding components for G2 and G2(MP2) calculation.\n")

print("Single-point energy: MP2/6-311+G(3df,2p) [frozen core]")

psi4.set_options({
    "basis": "6-311+G(3df,2p)",
    "scf_type": "direct",
    "df_scf_guess": False,
    "mp2_type": "conv",
    "freeze_core": "true",
})

psi4.energy("MP2")
e_mp2_plus_3df2p = psi4.get_variable("MP2 TOTAL ENERGY")
psi4.core.clean()

# e_mp2_base = -76.263654158359
# e_mp2_plus = -76.274546774908
# e_mp2_2df = -76.298942654928
# This calculation can be avoided by rearranging the expressions.
# e_mp2_plus_2df
# e_mp2_plus_3df2p = -76.318108464496

# de_mp2_plus_2df
de_mp2_plus = e_mp2_plus - e_mp2_base
de_mp2_2df = e_mp2_2df - e_mp2_base

# de_1 = de_plus_2df + de_plus + de_2df
# de_2 = e_mp2_plus_3df2p - e_mp2_plus_2df
# de = de_1 + de_2
de = e_mp2_plus_3df2p - e_mp2_2df - e_mp2_plus + e_mp2_base

# Calculate the number of valence pairs.
npairs = min(nalpha, nbeta)
e_hlc_2 = 1.14 * npairs / 1000

g2_0k = g1_0k + de + e_hlc_2
g2_energy = g2_0k - zpe + du
g2_enthalpy = g2_0k - zpe + dh
g2_free_energy = g2_0k - zpe + dg

# G2(MP2) bits

de_mp2 = e_mp2_plus_3df2p - e_mp2_base
g2mp2_0k = (e_step6_qci + e_step3) + de_mp2 + e_hlc_1 + e_hlc_2 + e_zpe

g2mp2_energy = g2mp2_0k - zpe + du
g2mp2_enthalpy = g2mp2_0k - zpe + dh
g2mp2_free_energy = g2mp2_0k - zpe + dg

print("e(Delta-G2)=              {:10.6f}".format(de))
print("E(G2-Empiric)=               {:10.6f}".format(e_hlc_2))
print("G2(0 K)=                  {:10.6f}".format(g2_0k))
print("G2 Energy=                   {:10.6f}".format(g2_energy))
print("G2 Enthalpy=              {:10.6f}".format(g2_enthalpy))
print("G2 Free Energy=              {:10.6f}".format(g2_free_energy))
print("DE(MP2)=                  {:10.6f}".format(de_mp2))
print("G2MP2(0 K)=               {:10.6f}".format(g2mp2_0k))
print("G2MP2 Energy=                {:10.6f}".format(g2mp2_energy))
print("G2MP2 Enthalpy=           {:10.6f}".format(g2mp2_enthalpy))
print("G2MP2 Free Energy=           {:10.6f}".format(g2mp2_free_energy))
