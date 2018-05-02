Molecular Dynamics Integrators
==============================

This directory contains a reference implementation of a Molecular Dynamics (MD)
simulation performed using the Velocity-Verlet integration scheme and Psi4
for computing gradients. Directory contents:

- `md_helper.py`: A collection of helper functions for performing MD simulations using Psi4
- `md_prog.py`: A MD program propagating the dynamics of a hydrogen molecule at the HF/3-21G level of theory

#### Helper Functions
- `md_trajectories()`: Creates complete trajectory from all simulation snapshots
- `integrator()`: Propagates MD trajectory via Velocity-Verlet integrator
- `get_forces()`: Uses Psi4 to obtain atomic forces for propagation

### References:
- [[Attig:2004](https://books.google.com/books/about/Computational_Soft_Matter_from_Synthetic.html?id=IG8rAwAACAAJ)] "Computational Soft Matter: From Synthetic Polymers to Proteins: Lecture Notes," N. Attig, K. Binder, H. Grubmuller, and K. Kremer (Eds.), NIC Series, Vol. 23, John von Neumann Institute for Computing (NIC), Julich Supercomputing Centre, Julich, Germany. (2004)

