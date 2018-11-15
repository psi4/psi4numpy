import psi4
import resp_driver
import resp_helper
import numpy as np

# Initialize two different conformations of ethanol
geometry = """C    0.00000000  0.00000000  0.00000000
C    1.48805540 -0.00728176  0.39653260
O    2.04971655  1.37648153  0.25604810
H    3.06429978  1.37151670  0.52641124
H    1.58679428 -0.33618761  1.43102358
H    2.03441010 -0.68906454 -0.25521028
H   -0.40814044 -1.00553466  0.10208540
H   -0.54635470  0.68178278  0.65174288
H   -0.09873888  0.32890585 -1.03449097
"""
mol1 = psi4.geometry(geometry)
mol1.update_geometry()
mol1.set_name('conformer1')

geometry = """C    0.00000000  0.00000000  0.00000000
C    1.48013500 -0.00724300  0.39442200
O    2.00696300  1.29224100  0.26232800
H    2.91547900  1.25572900  0.50972300
H    1.61500700 -0.32678000  1.45587700
H    2.07197500 -0.68695100 -0.26493400
H   -0.32500012  1.02293415 -0.30034094
H   -0.18892141 -0.68463906 -0.85893815
H   -0.64257065 -0.32709111  0.84987482
"""
mol2 = psi4.geometry(geometry)
mol2.update_geometry()
mol2.set_name('conformer2')

molecules = [mol1, mol2]

# Specify intermolecular constraints
intermolecular_constraint = {'EQUAL': [[[1, range(1, 10)], [2, range(1, 10)]]]}

# Specify options
options1 = {'N_VDW_LAYERS'       : 4,
           'VDW_SCALE_FACTOR'   : 1.4,
           'VDW_INCREMENT'      : 0.2,
           'VDW_POINT_DENSITY'  : 1.0,
           'resp_a'             : 0.0005,
           'RESP_B'             : 0.1,
           'restraint'          : True,
           'ihfree'             : False,
           'WEIGHT'             : 1,
           }
options2 = {'WEIGHT': 1}
options = [options1, options2]

# Call for first stage fit
charges1 = resp_driver.resp(molecules, options, intermolecular_constraint)

print("Restrained Electrostatic Potential Charges")
print(charges1[0][1])
# Reference Charges are generates with the resp module of Ambertools
# Grid and ESP values are from this code with Psi4
reference_charges1 = np.array([-0.149134, 0.274292, -0.630868,  0.377965, -0.011016,
                               -0.009444,  0.058576,  0.044797,  0.044831])
print("Reference RESP Charges")
print(reference_charges1)
print("Difference")
print(charges1[0][1]-reference_charges1)
print("Example works?")
assert np.allclose(charges1[0][1], reference_charges1, atol=2e-5)

# Add constraint for atoms fixed in second stage fit
stage2 = resp_helper.helper_stage2()
for mol in range(len(molecules)):
    stage2.set_stage2_constraint(molecules[mol], charges1[mol][1], options[mol], cutoff=1.2)
    options[mol]['grid'] = '%i_%s_grid.dat' %(mol+1, molecules[mol].name())
    options[mol]['esp'] = '%i_%s_grid_esp.dat' %(mol+1, molecules[mol].name())
    options[0]['resp_a'] = 0.001
    molecules[mol].set_name('conformer' + str(mol+1) + '_stage2')

# Add intermolecular constraints
stage2.stage2_intermolecular_constraint(molecules, cutoff=1.2)

# Call for second stage fit
charges2 = resp_driver.resp(molecules, options, stage2.intermolecular_constraint)
print("\nStage Two\n")
print("RESP Charges")
print(charges2[0][1])
reference_charges2 = np.array([-0.079853, 0.253918, -0.630868, 0.377965, -0.007711,
                               -0.007711, 0.031420,  0.031420, 0.031420])
print("Reference RESP Charges")
print(reference_charges2)
print("Difference")
print(charges2[0][1]-reference_charges2)
print("Example works?")
assert np.allclose(charges2[0][1], reference_charges2, atol=2e-5)
