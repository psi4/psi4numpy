"""
A Psi4 input script to compute TDHF linear response. As a note this is, by far,
not the most efficiently algorithm, but certainly the most verbose.

References:
- TDHF equations and algorithms taken from [Amos:1985:2186] and [Helgaker:2000]
- Gauss-Legendre integration from [Amos:1985:2186] and [Jiemchooroj:2006:124306]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-09-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set memory & output file
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Set molecule to dimer
mol = psi4.geometry("""
Be  0  0  0
symmetry c1
""")

psi4.set_options({"scf_type": "out_of_core",
                  "basis": "aug-cc-pVTZ",
                  "e_convergence": 1e-8,
                  "d_convergence": 1e-8})

t = time.time()
scf_e, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF took                %5.3f seconds' % ( time.time() - t))

Co = wfn.Ca_subset("AO", "OCC")
Cv = wfn.Ca_subset("AO", "VIR")
epsilon = np.asarray(wfn.epsilon_a())

nbf = wfn.nmo()
ndocc = wfn.nalpha()
nvir = nbf - ndocc
nov = ndocc * nvir
print('')
print('Ndocc: %d' % ndocc)
print('Nvir:  %d' % nvir)
print('Nrot:  %d' % nov)
print('')

eps_v = epsilon[ndocc:]
eps_o = epsilon[:ndocc]

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
I = mints.ao_eri()
v_ijab = np.asarray(mints.mo_transform(I, Co, Co, Cv, Cv))
v_iajb = np.asarray(mints.mo_transform(I, Co, Cv, Co, Cv))
Co = np.asarray(Co)
Cv = np.asarray(Cv)
print('Integral transform took %5.3f seconds\n' % ( time.time() - t))

# Grab perturbation tensors in MO basis
tmp_dipoles = mints.so_dipole()
dipoles_xyz = []
for num in range(3):
    Fso = np.asarray(tmp_dipoles[num])
    Fia = (Co.T).dot(Fso).dot(Cv)
    Fia *= -2
    dipoles_xyz.append(Fia)

# Build orbital-Hessian
t = time.time()
E1  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
E1 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
E1 += 4 * v_iajb
E1 -= v_ijab.swapaxes(1, 2)
E1 -= v_iajb.swapaxes(0, 2)
E1 *= 4


# Since we are time dependent we need to build the full Hessian:
# | A B |      | D  S | |  x |   |  b |
# | B A |  - w | S -D | | -x | = | -b |

# Build A and B blocks
A11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
A11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
A11 += 2 * v_iajb
A11 -= v_ijab.swapaxes(1, 2)
A11 *= 2

B11  = -2 * v_iajb
B11 += v_iajb.swapaxes(0, 2)
B11 *= 2

# The blocks A - B should be equal to E1 / 2
print('A11 - B11 == E1 / 2?',  np.allclose(A11 - B11, E1/2))
print('')

# Reshape and jam it together
A11.shape = (nov, nov)
B11.shape = (nov, nov)

Hess1 = np.hstack((A11, B11))
Hess2 = np.hstack((B11, A11))
Hess = np.vstack((Hess1, Hess2))

S11 = np.zeros_like(A11)
D11 = np.zeros_like(B11)
S11[np.diag_indices_from(S11)] = 2

S1 = np.hstack((S11, D11))
S2 = np.hstack((D11, -S11))
S = np.vstack((S1, S2))
print('Hessian formation took  %5.3f seconds\n' % ( time.time() - t))


Hess = Hess.astype(np.complex)
S = S.astype(np.complex)

dip_x = dipoles_xyz[0].astype(np.complex)
B = np.hstack((dip_x.ravel(), -dip_x.ravel()))

C6 = np.complex(0, 0)
hyper_polar = np.complex(0, 0)
leg_points = 10
fdds_lambda = 0.30
print('     Omega      value     weight        sum')

# Integrate over time use a Gauss-Legendre polynomial.
# Shift from [-1, 1] to [0, inf) by the transform  (1 - x) / (1 + x)
for point, weight in zip(*np.polynomial.legendre.leggauss(leg_points)):
    if point != 0:
        omega = fdds_lambda * (1.0 - point) / (1.0 + point)
        lambda_scale = ( (2 * fdds_lambda) / (point + 1) ** 2)
    else:
        omega = 0
        lambda_scale = 0

    Hw = Hess - S * complex(0, omega)

    Z =  np.linalg.solve(Hw, B)
    value = -np.vdot(Z, B)

    if abs(value.imag) > 1.e-13:
        print('Warning value of imaginary part is large', value)

    C6 += (value ** 2) * weight * lambda_scale
    hyper_polar += value * weight * lambda_scale
    print('% .3e % .3e % .3e % .3e' % (omega, value.real, weight, weight*value.real))

C6 *= 3.0 / np.pi
print('\nFull C6 Value: %s' % str(C6))

# We can solve static using the above with omega = 0. However a simpler way is
# just to use the reduced form:
dip_x = dip_x.ravel()
static_polar = 2 * np.dot(dip_x, np.linalg.solve(A11 - B11, dip_x))

print('\nComputed values:')
print('Alpha                 % 10.5f' % static_polar.real)
print('C6                    % 10.5f' % C6.real)

print('\nBenchmark values:')
print('C6 He  Limit          % 10.5f' % 1.376)
print('C6 Li+ Limit          % 10.5f' % 0.076)
print('C6 Be  Limit          % 10.5f' % 282.4)
