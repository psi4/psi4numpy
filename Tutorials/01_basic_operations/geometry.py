import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()
mol.print_out()

psi4.core.set_global_option("BASIS", "cc-pVDZ")
