Moller-Plesset Perturbation Theory
=================
Moller-Plesset perturbation theory is a post-SCF quantum chemistry method which
adds dynamic electron correlation to improve upon SCF level wavefunctions. This
method is a workhose of quantum chemistry as it has proved to be remarkably
accurate considering its overall cost and is the basis for many theories built
ontop of this result.

### Included Reference Implementations
 - `MP2.py`: Second-order Moller-Plesset perturbation theory
 - `DF-MP2.py`: MP2 utilizing density-fitting to reduce its overall cost.
 - `sDF-MP2.py`: Stochastic orbital MP2 utilizing density-fitting (sDF-MP2) to further reduce the computational cost of DF-MP2.
 - `MP3.py`: Third-order Moller-Plesset perturbation theory
 - `MP3-SO.py`: MP3 in spin-orbital formalism that simplifies the equations and provides contrast to the spin-summed version above.
 - `MPn.py`: An example on how to automate the contraction of higher order MP theory.
 - `MP2_Gradient.py`: Calculation of a nuclear gradient at the MP2 level of theory.

### References
 1) The original paper that started it all: "Note on an Approximation Treatment for Many-Electron Systems"
    - [[Moller:1934:618](https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618)] C. Møller and M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
 2) The Laplace-transformation in MP theory: "Minimax approximation for the decomposition of energy denominators in Laplace-transformed Møller–Plesset perturbation theories"
    - [[Takasuka:2008:044112](http://aip.scitation.org/doi/10.1063/1.2958921)] A. Takatsuka, T. Siichiro, and W. Hackbusch, *J. Phys. Chem.*, **129**, 044112 (2008)
 3) Equations used in reference implementations:
    - [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Courier Corporation, 1996.
 4) sDF-MP2 reference:
    - [[Takeshita:2017:4605](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00343)] T.Y. Takeshita, W.A. de Jong, D. Neuhauser, R. Baer and E. Rabani, *J. chem. Theory Comput.* **13**, 4605 (2017)
