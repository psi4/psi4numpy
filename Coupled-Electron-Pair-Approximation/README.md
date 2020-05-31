Coupled Electron Pair Approximation
======================

Reference implementations for various truncations of spin-orbital--based CEPA method. 

The following codes are available:
- `LCCD.py`: An implementation of linearized CCD theory with DIIS.
This is also commonly known as CEPA(0) without singles and also DMBPT(infinity), among other names.
- `LCCSD.py`: An implementation of linearized CCSD theory with DIIS.
This is also commonly known as CEPA(0) with singles.
- `OLCCD.py`: An implementation of orbital optimized linearized CCD theory with DIIS.
- `DF-LCCD.py`: An implementation of linearized CCD theory with DIIS and the density-fitting approximation.
- `DF-LCCSD.py`: An implementation of linearized CCSD theory with DIIS and the density-fitting approximation.

Helper programs:
- `DSD.py`: DIIS code capable of doing a combined extrapolation of multiply vectors.
- `integrals.py`: Boilerplate code to generate the starting integrals. 

### References:
- CEPA Overview:
    1. [[Ahlrichs:1979:31](https://www.sciencedirect.com/science/article/pii/0010465579900675)] R. Ahlrichs, *Comp. Phys. Comm.* **17**, 31 (1979)
- LCCSD Overview:
    1. [[Taube:2009:144112](https://aip.scitation.org/doi/10.1063/1.3115467)] A. G. Taube, R. J. Bartlett, *J. Chem. Phys.* **130** 144112 (2009).
- Coupled Cluster Equations:
    1. [[Crawford:2000:33](https://onlinelibrary.wiley.com/doi/10.1002/9780470125915.ch2)] T. D. Crawford, H. F. Schaefer III, *Reviews in Computational Chemistry* **14**, 33 (2000)
- Orbital-Optimized LCCD:
    1. [[Bozkaya:2013:054104](https://aip.scitation.org/doi/10.1063/1.4816628)] U. Bozkaya, C. D. Sherrill, *J. Chem. Phys.* **139** 054104 (2013).
