# A simple Psi4 input script to compute a Wilson B matrix
# edited with jupyter
# Created by: Rollin A. King
# Date: 7/27/17
# License: GPL v3.0

import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('1 GB')
psi4.core.set_output_file('output.dat',False)

import optking
# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set some options
psi4.set_options({"basis": "cc-pvdz",
                  "scf_type": "pk",
                  "e_convergence": 1e-8})

mol.update_geometry()
xyz = np.array( mol.geometry() )
print('Cartesian geometry')
print(xyz)

# Create some internal coordinates for water.
# Numbering starts with atom 0.
from optking import stre,bend
r1 = stre.STRE(0,1)
r2 = stre.STRE(0,2)
theta = bend.BEND(1,0,2)
intcos = [r1,r2,theta]

print("%15s%15s" % ('Coordinate', 'Value'))
for I in intcos:
  print("%15s = %15.8f %15.8f" % (I, I.q(xyz), I.qShow(xyz)))

#from optking import printTools
optking.printInit()

from optking import optParams as op
userOptions = {'print' : 3}
op.Params = op.OPT_PARAMS(userOptions)
optking.printTools.print_opt("\tParameters from optking.optimize\n")
optking.printTools.print_opt( str(op.Params) )

# displacements to be made in au
dq = np.array( [0.0, 0.2, 0.0], float)

# forces - only used for printing here
fq = np.array( [0.0, 0.0, 0.0], float)

from optking import displace
displace.displace(intcos, xyz, dq, fq)

print('Final Displaced Cartesian geometry')
print(xyz)

# Check result
print("%15s%15s" % ('Coordinate', 'Value'))
for I in intcos:
  print("%15s = %15.8f %15.8f" % (I, I.q(xyz), I.qShow(xyz)))

