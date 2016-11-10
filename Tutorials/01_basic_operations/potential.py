import numpy as np
import psi4

# Set the output file to avoid most of the Psi4 printing
psi4.core.set_output_file("potential.out", False)

mol_string = """
He
--
He 1 **R**
"""

# Compute the cp correct MP2 energy for a list of distances
energies = []
distances = np.linspace(2.5, 7, 6)
for dist in distances:
    tmp_mol_string = mol_string.replace("**R**", str(dist))
    mol = psi4.geometry(tmp_mol_string)
    
    energy = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp')
    print('Completed distance %4.2f' % dist)
    energies.append([dist, energy])

energies = np.array(energies)

# Distances in angstrom, energies in wavenumbers
energies[:, 1] *= 219474.63067

# Fit the data in a linear leastsq way to a -12, -6 polynomial
x = np.power(energies[:, 0].reshape(-1, 1), [-12, -6])
y = energies[:, 1] 
coefs = np.linalg.lstsq(x, y)[0]

fit_energies = np.dot(x, coefs)
print('Best fit: %1.5e * R^-12 + %1.5e * R^-6' % (coefs[0], coefs[1]))

print('Distances:     ' + ', '.join('% 5.3f' % x for x in energies[:, 0]))
print('Energies:      ' + ', '.join('% 5.3f' % x for x in energies[:, 1]))
print('Fit energies:  ' + ', '.join('% 5.3f' % x for x in fit_energies))

# Note that the fitting is not balanced as seen by the error in the long range
# points. A good exercise would be to use a relative error metric in the
# fitting.

