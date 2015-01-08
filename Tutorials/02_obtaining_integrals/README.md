One of the most common operations in psi4numpy is to obtain integrals from the psi4 program and convert them into numpy arrays.
In psi4 there is a convenient helper function that build dense arrays of integrals called ```MintsHelper```.

First, lets discuss a few pieces that all psi4numpy code must have.
All psi4numpy code must do the following import numpy, specifying a molecule, and pick a basis set.
A short example is shown here:
'''python
import numpy as np

molecule mol {
O
H 1 1.1
H 1 1.1 2 105
}

set {
 basis = sto-3g
}
```
From this point forward all code will assume that we have at least these elements.

Fortunately, mintshelper and numpy both uses c-contigous strided arrays.
The benefit of this is arbitrary dimension psi4 matrices can be seamlessly passed to numpy without copying data.
Any values computed with ```MintsHlper``` automatically returns a psi4 matrix or vector class. 
These classes have a built in attribute called __array_interface__ which provides the information numpy requires to build numpy arrays.
For a more in-depth and tehcnical explination please see [here](...).

Lets start with a practicle example where we construct the atomic orbital kinetic energy integrals and set this equivalent to T:
```python
mints = MintsHelper()

T = mints.ao_kinetic()
print(T)
# <psi4.Matrix object at 0x104089c08>

np_T = np.array(T)
print(np_T)

[[  2.90031999e+01  -1.68010939e-01  -1.95597464e-17   0.00000000e+00
    0.00000000e+00  -8.41638490e-03  -8.41638490e-03]
...
 [ -8.41638490e-03   7.05173444e-02   1.13632016e-01   0.00000000e+00
    1.48088126e-01  -4.40920221e-03   7.60031884e-01]]
```
This is the basic procedure for converting any tensor like quantity in psi4 to a numpy array.
A full list of possible integrals can be found by typing ```help(MintsHelper())``` inside a psi4 input script.
The most commonly used are listed below:
 - ao_overlap(): Atomic orbital overlap integrals 
 - ao_kinetic(): Atomic orbital kinetic integrals
 - ao_potential(): Atomic orbital potential integrals
 - ao_eri(): Atomic orbital electron repulsion integrals

## Current
A special note on ao_eri():
Currently in the public version of psi4 only 2D arrays are support therefore the ao_eri() is returned as a two-index tensor.
Again, all tensors are strided arrays so the data is correctly ordered and only the strides of the array need to be changed:
'''python
mints = MintsHelper()
S = np.array(mints.ao_overlap())
nbf = S.shape[0]
I = np.array(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
'''

## Beta
In the beta version: n-dimensional arrays are built in so the ERI tensor is automatically returned as a 4-index tensor.
No reshaping necessary!


