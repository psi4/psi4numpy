## psi4 input scripts

One of the basic principles of psi4 input scripts is that they are, in essence, python scripts.
The primary reason that psi4 input scripts are called "psithon" scripts is that there is additional functionality and parsing that convention python scripts do not have.
On top of this parsing the psithon adds additional modules that are fairly transparent to the user, these are:
 - psi4 - The main psi4 C++ library that computes integrals, computes methods, etc
 - aliases - A python library that calls psi4 to compute various quantities, a primary example of this is the "energy" function.
 - Other utility modules are loaded, but we can safely avoid discussing them for now.

To begin with lets write a simple input script (```geometry.py```) that does nothing but load a geometry:
```python
memory 2 GB
molecule mol {
O
H 1 1.1
H 1 1.1 2 104
}

set {
basis sto-3g
}
```
Running Psi4 verbose mode will print out the actual python script itself belwo the input script.
This can be accomplished by:  ```psi4 -v input.dat``` and will yield:
```python
from psi4 import *
from p4const import *
from p4util import *
from molutil import *
from aliases import *
psi4_io = psi4.IOManager.shared_object()
psi4.set_memory(2000000000)

mol = geometry("""
O
H 1 1.1
H 1 1.1 2 104
""","mol")
psi4.IO.set_default_namespace("mol")
psi4.set_global_option("basis", "sto-3g")
```
We can see both the additional parsing that is done to the psithon scripts in addition to the additional modules that are loaded.
The main point here is that all psi4 input scripts are simply python input scripts with additional utilities pre-loaded and expression handeling.
Psi4 can be compiled as a shared object that can be imported directly into a generic python scripts; however, this is currently fairly complicated and avoid for now.

## Using numpy in psi4

For some reason we have the sudden urge to compute the MP2 energy between two helium molecules and fit those points to a Leonnard-Jones-like potential using numpy.
We can safely treat the output from ```cp('MP2')``` (compute the cp corrected MP2 interaction energy) as a normal python float and import numpy the same way as a generic python script:
```python
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
print('Best fit: %1.5e * R^-12 + %1.5e * R^-6' % (coefs[0], coefs[1]))

# Compute fitted energies
fit_energies = np.dot(x, coefs)

# Print the results in a nice way
print('Distances:     ' + ', '.join('% 2.3f' % x for x in energies[:, 0]))
print('Energies:      ' + ', '.join('% 2.3f' % x for x in energies[:, 1]))
print('Fit energies:  ' + ', '.join('% 2.3f' % x for x in fit_energies))
```

If the above is difficult to follow one of two things have happened.
Either the code is confusing or I have not explained something very well.
If it is the former I would encourage you to try and debug the code just like a generic python script if it is the latter (frankly the more likely of the two) I would strongly encourage you to email me so I can improve this.



