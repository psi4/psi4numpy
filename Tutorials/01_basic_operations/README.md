## Psi4 scripts

Before starting this ensure that you have read the [NumPy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html).

To begin let us build a water molecule and print the resulting geometry to the screen.

```python
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()
mol.print_out()
```

When a Psi4 molecule is first built using `psi4.geometry` it is in an unfished
state as a user may wish to tweak the molecule. This can be solved by calling
`update_geometry`. By default Psi4 will print its output to `stdout` (the screen
in most cases) and running this script will yeild a string version of our
molecule:

```
    Molecular point group: c2v
    Full point group: C2v

    Geometry (in Angstrom), charge = 0, multiplicity = 1:

       Center              X                  Y                   Z               Mass
    ------------   -----------------  -----------------  -----------------  -----------------
           O          0.000000000000     0.000000000000    -0.075791843589    15.994914619560
           H          0.000000000000    -0.866811828967     0.601435779270     1.007825032070
           H          0.000000000000     0.866811828967     0.601435779270     1.007825032070
```

A more advanced example where we construct a potenital energy curve and fit it
to a Lennard-Jones can be found in `potential.py`.



