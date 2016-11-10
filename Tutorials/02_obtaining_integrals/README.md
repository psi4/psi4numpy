## Obtaining integrals
One of the most common operations in Psi4NumPy is obtaining integrals from the Psi4 program.
In Psi4 there is a convenient helper function that builds dense arrays (required for numpy to work) of integrals called ```MintsHelper```.

All Psi4NumPy code should begin by specifying a molecule and picking a basis set.
A short example is shown here:
```python
mol = Psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 105
""")

basis = Psi4.core.BasisSet.build(mol, target="STO-3G")
```
From this point forward all code will assume that we have at least these elements.

Fortunately, MintsHelper and numpy both uses c-contiguous strided arrays.
The benefit of this is arbitrary dimension Psi4 matrices can be seamlessly passed to numpy without copying data.
Any values computed with ```MintsHlper``` automatically returns a Psi4 matrix or vector class. 
These classes have a built in attribute called __array_interface__ which provides the information numpy requires to build numpy arrays.
For a more in-depth and technical explanation please see [here](http://docs.scipy.org/doc/numpy/reference/arrays.interface.html).

Lets start with a practicle example where we construct the atomic orbital kinetic energy integrals and set this equivalent to ```T``` and convert this matrix to a numpy array ```np_T```:
```python
mints = psi4.core.MintsHelper(basis)

T = mints.ao_kinetic()
print(T)
# <psi4.core.Matrix object at 0x104089c08>

np_T = np.array(T)
print(np_T)

# [[  2.90031999e+01  -1.68010939e-01  -1.95597464e-17   0.00000000e+00
#     0.00000000e+00  -8.41638490e-03  -8.41638490e-03]
# ...
#  [ -8.41638490e-03   7.05173444e-02   1.13632016e-01   0.00000000e+00
#     1.48088126e-01  -4.40920221e-03   7.60031884e-01]]
```
This is the basic procedure for converting any tensor like quantity in Psi4 to a numpy array.
A full list of possible integrals can be found by typing ```help(psi4.core.MintsHelper())``` inside a Psi4 input script.
The most commonly used are listed below:
 - ao_overlap(): Atomic orbital overlap integrals 
 - ao_kinetic(): Atomic orbital kinetic integrals
 - ao_potential(): Atomic orbital potential integrals
 - ao_eri(): Atomic orbital electron repulsion integrals

