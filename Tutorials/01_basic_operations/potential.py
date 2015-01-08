import numpy as np

memory 2 GB
molecule mol {
He
--
He 1 R
}

set {
basis aug-cc-pVDZ
}

energies = []
distances = np.linspace(2.5, 7, 10)
for dist in distances:
    mol.R = dist
    energy = cp('MP2')
    energies.append([dist, energy])

energies = np.array(energies)

# Distances in angstrom, energies in wavenumbers
energies[:, 1] *= 219474.63067

# Fit the data in a linear leastsq way
x = np.power(energies[:, 0].reshape(-1, 1), [-12, -6])
y = energies[:, 1] 
coefs = np.linalg.lstsq(x, y)[0]

fit_energies = np.dot(x, coefs)
print('Best fit: %1.5e * R^-12 + %1.5e * R^-6' % (coefs[0], coefs[1]))

print('Distances:     ' + ', '.join('% 2.3f' % x for x in energies[:, 0]))
print('Energies:      ' + ', '.join('% 2.3f' % x for x in energies[:, 1]))
print('Fit energies:  ' + ', '.join('% 2.3f' % x for x in fit_energies))

