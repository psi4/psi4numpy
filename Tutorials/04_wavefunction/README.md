## RHF Wavefunction

Instead of writing a SCF code at the beginning of every psi4numpy input script we can request that psi4 compute the SCF energy.
For most methods, Psi4 builds a "wavefunction" object that contains all of the relevant quantities in the calculation and these can be accessed after the computation is complete.
For example, the SCF wavefunction object holds the following:
 - ```energy()```: The energy of the current method.
 - ```nalpha()```: The number of alpha electrons.
 - ```nbeta()```: The number of beta electrons.
 - ```nirrep()```: The number of irreducible representations (number of symmetry elements).
The following objects can utilize molecular symmetry:
 - ```ndoccpi()```: The number of doubly occupied orbitals.
 - ```Ca()```: The orbitals (C matrix).
 - ```Da()```: The density matrix.
 - ```Fa()```: The Fock matrix.
 - ```epsilon_a()```: The current eigenvalues (epsilon vector).

The full list is quite extensive; however, this likely comprises the most utilized functions.
It should be noted that the ```a``` stands for alpha and conversely the beta quantities can be accessed with the letter ```b```.
For now lets ensure that all computations have C1 symmetry;
molecular symmetry can be utilized in psi4numpy computations, but adds significant complexity to the code.



