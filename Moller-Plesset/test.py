import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

mol.update_geometry()

options = {'BASIS':'STO-3G',
           'SCF_TYPE':'PK',
           'MP2_TYPE':'CONV',
           'E_CONVERGENCE':1e-12,
           'D_CONVERGENCE':1e-12,
           'print':1}

psi4.set_options(options)

psi4.gradient("MP2")

