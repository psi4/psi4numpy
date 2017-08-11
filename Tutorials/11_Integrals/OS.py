from collections import namedtuple
import numpy as np
from scipy import linalg as la
import psi4 as p4
import configparser
import math
import sys

config = configparser.ConfigParser()
config.read('Options.ini')

DirectionalResults = namedtuple('DirectionalResults', ['x', 'y', 'z'])


def os_recursion(PA, PB, alpha, AMa, AMb):
    if len(PA) != 3 or len(PB) != 3:
        raise "PA and PB must be xyz coordinates."

    # Allocate x, y, and z matrices
    x = np.zeros((AMa + 1, AMb + 1))
    y = np.zeros((AMa + 1, AMb + 1))
    z = np.zeros((AMa + 1, AMb + 1))

    # Perform recursion
    oo2a = 1.0 / (2.0 * alpha)

    # Initial conditions
    x[0, 0] = y[0, 0] = z[0, 0] = 1.0

    # Upward recursion in b for a = 0
    if AMb > 0:
        x[0, 1] = PB[0]
        y[0, 1] = PB[1]
        z[0, 1] = PB[2]

    for b in range(1, AMb):
        x[0, b + 1] = PB[0] * x[0, b] + b * oo2a * x[0, b - 1]
        y[0, b + 1] = PB[1] * y[0, b] + b * oo2a * y[0, b - 1]
        z[0, b + 1] = PB[2] * z[0, b] + b * oo2a * z[0, b - 1]

    # Upward recursion in a for all b's
    if AMa > 0:
        x[1, 0] = PA[0]
        y[1, 0] = PA[1]
        z[1, 0] = PA[2]

        # Need to check this range
        for b in range(1, AMb + 1):
            x[1, b] = PA[0] * x[0, b] + b * oo2a * x[0, b - 1]
            y[1, b] = PA[1] * y[0, b] + b * oo2a * y[0, b - 1]
            z[1, b] = PA[2] * z[0, b] + b * oo2a * z[0, b - 1]

        for a in range(1, AMa):
            x[a + 1, 0] = PA[0] * x[a, 0] + a * oo2a * x[a - 1, 0]
            y[a + 1, 0] = PA[1] * y[a, 0] + a * oo2a * y[a - 1, 0]
            z[a + 1, 0] = PA[2] * z[a, 0] + a * oo2a * z[a - 1, 0]

            for b in range(1, AMb + 1):
                x[a + 1, b] = PA[0] * x[a, b] + a * oo2a * x[a - 1, b] + b * oo2a * x[a, b - 1]
                y[a + 1, b] = PA[1] * y[a, b] + a * oo2a * y[a - 1, b] + b * oo2a * y[a, b - 1]
                z[a + 1, b] = PA[2] * z[a, b] + a * oo2a * z[a - 1, b] + b * oo2a * z[a, b - 1]

    # Return the results
    return DirectionalResults(x, y, z)


class OverlapInt:
    def __init__(self, molecule, basis_a, basis_b):
        self.molecule = molecule
        self.basisA = basis_a
        self.basisB = basis_b

    def compute_shell_pair(self, nsa, nsb):
        Sa = self.basisA.shell(nsa)
        Sb = self.basisB.shell(nsb)

        AMa = Sa.am
        AMb = Sb.am

        NPrimA = Sa.nprimitive
        NPrimB = Sb.nprimitive

        A = np.zeros(3)
        B = np.zeros(3)

        A[0] = molecule.x(Sa.ncenter)
        A[1] = molecule.y(Sa.ncenter)
        A[2] = molecule.z(Sa.ncenter)
        B[0] = molecule.x(Sb.ncenter)
        B[1] = molecule.y(Sb.ncenter)
        B[2] = molecule.z(Sb.ncenter)

        AB2 = pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2) + pow(A[2] - B[2], 2)

        buffer = np.zeros(Sa.ncartesian * Sb.ncartesian)

        for PrimA in range(NPrimA):
            alphaA = Sa.exp(PrimA)
            coeffA = Sa.coef(PrimA)

            for PrimB in range(NPrimB):
                alphaB = Sb.exp(PrimB)
                coeffB = Sb.coef(PrimB)

                alpha = alphaA + alphaB
                ooa = 1.0 / alpha

                P = np.zeros(3)
                PA = np.zeros(3)
                PB = np.zeros(3)

                P[0] = (alphaA * A[0] + alphaB * B[0]) * ooa
                P[1] = (alphaA * A[1] + alphaB * B[1]) * ooa
                P[2] = (alphaA * A[2] + alphaB * B[2]) * ooa

                PA[0] = P[0] - A[0]
                PA[1] = P[1] - A[1]
                PA[2] = P[2] - A[2]

                PB[0] = P[0] - B[0]
                PB[1] = P[1] - B[1]
                PB[2] = P[2] - B[2]

                overlap_ss = math.exp(-alphaA * alphaB * AB2 * ooa) * math.sqrt(
                    math.pi * ooa) * math.pi * ooa * coeffA * coeffB

                print('i,j ', nsa, nsb, overlap_ss / (coeffA * coeffB))
                results = os_recursion(PA, PB, alpha, AMa, AMb)

                # Atomic orbital indexing
                AOab = 0

                for ii in range(AMa + 1):
                    la = AMa - ii
                    for jj in range(ii + 1):
                        ma = ii - jj
                        na = jj

                        for kk in range(AMb + 1):
                            lb = AMb - kk
                            for ll in range(kk + 1):
                                mb = kk - ll
                                nb = ll

                                x0 = results.x[la, lb]
                                y0 = results.y[ma, mb]
                                z0 = results.z[na, nb]

                                buffer[AOab] += overlap_ss * x0 * y0 * z0
                                AOab += 1

        buffer.shape = (Sa.ncartesian, Sb.ncartesian)
        return buffer

    def compute(self):
        """Returns a NumPy array containing the integrals."""
        S = np.zeros((self.basisA.nao(), self.basisB.nao()))
        nsA = self.basisA.nshell()
        nsB = self.basisB.nshell()

        for i in range(nsA):
            ni = self.basisA.shell(i).ncartesian
            firsti = self.basisA.shell(i).function_index

            for j in range(nsB):
                nj = self.basisB.shell(j).ncartesian
                firstj = self.basisB.shell(j).function_index

                # compute the pair
                buffer = self.compute_shell_pair(i, j)

                # for each integral that we received store it appropriately
                for p in range(ni):
                    for q in range(nj):
                        S[firsti + p, firstj + q] += buffer[p, q]

                # S[firsti:ni, firstj:nj] += buffer[0:ni, 0:nj]

        return S


class KineticInt:
    def __init__(self, molecule, basis_a, basis_b):
        self.molecule = molecule
        self.basisA = basis_a
        self.basisB = basis_b

    def compute_shell_pair(self, nsa, nsb):
        Sa = self.basisA.shell(nsa)
        Sb = self.basisB.shell(nsb)

        AMa = Sa.am
        AMb = Sb.am

        NPrimA = Sa.nprimitive
        NPrimB = Sb.nprimitive

        A = np.zeros(3)
        B = np.zeros(3)

        A[0] = molecule.x(Sa.ncenter)
        A[1] = molecule.y(Sa.ncenter)
        A[2] = molecule.z(Sa.ncenter)
        B[0] = molecule.x(Sb.ncenter)
        B[1] = molecule.y(Sb.ncenter)
        B[2] = molecule.z(Sb.ncenter)

        AB2 = pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2) + pow(A[2] - B[2],
                                                              2)

        buffer = np.zeros(Sa.ncartesian * Sb.ncartesian)

        for PrimA in range(NPrimA):
            alphaA = Sa.exp(PrimA)
            coeffA = Sa.coef(PrimA)

            for PrimB in range(NPrimB):
                alphaB = Sb.exp(PrimB)
                coeffB = Sb.coef(PrimB)

                alpha = alphaA + alphaB
                ooa = 1.0 / alpha

                P = np.zeros(3)
                PA = np.zeros(3)
                PB = np.zeros(3)

                P[0] = (alphaA * A[0] + alphaB * B[0]) * ooa
                P[1] = (alphaA * A[1] + alphaB * B[1]) * ooa
                P[2] = (alphaA * A[2] + alphaB * B[2]) * ooa

                PA[0] = P[0] - A[0]
                PA[1] = P[1] - A[1]
                PA[2] = P[2] - A[2]

                PB[0] = P[0] - B[0]
                PB[1] = P[1] - B[1]
                PB[2] = P[2] - B[2]

                overlap_ss = math.exp(
                    -alphaA * alphaB * AB2 * ooa) * math.sqrt(
                    math.pi * ooa) * math.pi * ooa * coeffA * coeffB

                (x, y, z) = os_recursion(PA, PB, alpha, AMa + 1, AMb + 1)

                # Atomic orbital indexing
                AOab = 0

                for ii in range(AMa + 1):
                    l1 = AMa - ii
                    for jj in range(ii + 1):
                        m1 = ii - jj
                        n1 = jj

                        for kk in range(AMb + 1):
                            l2 = AMb - kk
                            for ll in range(kk + 1):
                                m2 = kk - ll
                                n2 = ll

                                I1 = 0.0 if (l1 == 0 or l2 == 0) else x[l1 - 1, l2 - 1] * y[m1, m2] * z[n1, n2]
                                I2 = x[l1 + 1, l2 + 1] * y[m1, m2] * z[n1, n2]
                                I3 = 0.0 if l2 == 0 else x[l1 + 1, l2 - 1] * y[m1, m2] * z[n1, n2]
                                I4 = 0.0 if l1 == 0 else x[l1 - 1, l2 + 1] * y[m1, m2] * z[n1, n2]
                                Ix = 0.5 * l1 * l2 * I1 + 2.0 * alphaA * alphaB * I2 - alphaA * l2 * I3 - l1 * alphaB * I4

                                I1 = 0.0 if (m1 == 0 or m2 == 0) else x[l1, l2] * y[m1-1, m2-1] * z[n1, n2]
                                I2 = x[l1, l2] * y[m1+1, m2+1] * z[n1, n2]
                                I3 = 0.0 if m2 == 0 else x[l1, l2] * y[m1+1, m2-1] * z[n1, n2]
                                I4 = 0.0 if m1 == 0 else x[l1, l2] * y[m1-1, m2+1] * z[n1, n2]
                                Iy = 0.5 * m1 * m2 * I1 + 2.0 * alphaA * alphaB * I2 - alphaA * m2 * I3 - m1 * alphaB * I4

                                I1 = 0.0 if n1 == 0 or n2 == 0 else x[l1, l2] * y[m1, m2] * z[n1-1, n2-1]
                                I2 = x[l1, l2] * y[m1, m2] * z[n1+1, n2+1]
                                I3 = 0.0 if n2 == 0 else x[l1, l2] * y[m1, m2] * z[n1+1, n2-1]
                                I4 = 0.0 if n1 == 0 else x[l1, l2] * y[m1, m2] * z[n1-1, n2+1]
                                Iz = 0.5 * n1 * n2 * I1 + 2.0 * alphaA * alphaB * I2 - alphaA * n2 * I3 - n1 * alphaB * I4

                                buffer[AOab] += overlap_ss * (Ix + Iy + Iz)
                                AOab += 1

        buffer.shape = (Sa.ncartesian, Sb.ncartesian)
        return buffer

    def compute(self):
        """Returns a NumPy array containing the integrals."""
        S = np.zeros((self.basisA.nao(), self.basisB.nao()))
        nsA = self.basisA.nshell()
        nsB = self.basisB.nshell()

        for i in range(nsA):
            ni = self.basisA.shell(i).ncartesian
            firsti = self.basisA.shell(i).function_index

            for j in range(nsB):
                nj = self.basisB.shell(j).ncartesian
                firstj = self.basisB.shell(j).function_index

                # compute the pair
                buffer = self.compute_shell_pair(i, j)

                # for each integral that we received store it appropriately
                for p in range(ni):
                    for q in range(nj):
                        S[firsti + p, firstj + q] += buffer[p, q]

        return S


class DipoleInt:
    def __init__(self, molecule, basis_a, basis_b):
        self.molecule = molecule
        self.basisA = basis_a
        self.basisB = basis_b

    def compute_shell_pair(self, nsa, nsb):

        Sa = self.basisA.shell(nsa)
        Sb = self.basisB.shell(nsb)

        AMa = Sa.am
        AMb = Sb.am

        NPrimA = Sa.nprimitive
        NPrimB = Sb.nprimitive

        A = np.zeros(3)
        B = np.zeros(3)

        A[0] = molecule.x(Sa.ncenter)
        A[1] = molecule.y(Sa.ncenter)
        A[2] = molecule.z(Sa.ncenter)
        B[0] = molecule.x(Sb.ncenter)
        B[1] = molecule.y(Sb.ncenter)
        B[2] = molecule.z(Sb.ncenter)

        AB2 = pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2) + pow(A[2] - B[2],
                                                              2)

        buffer = np.zeros((3, Sa.ncartesian * Sb.ncartesian))

        for PrimA in range(NPrimA):
            alphaA = Sa.exp(PrimA)
            coeffA = Sa.coef(PrimA)

            for PrimB in range(NPrimB):
                alphaB = Sb.exp(PrimB)
                coeffB = Sb.coef(PrimB)

                alpha = alphaA + alphaB
                ooa = 1.0 / alpha

                P = np.zeros(3)
                PA = np.zeros(3)
                PB = np.zeros(3)

                P[0] = (alphaA * A[0] + alphaB * B[0]) * ooa
                P[1] = (alphaA * A[1] + alphaB * B[1]) * ooa
                P[2] = (alphaA * A[2] + alphaB * B[2]) * ooa

                PA[0] = P[0] - A[0]
                PA[1] = P[1] - A[1]
                PA[2] = P[2] - A[2]

                PB[0] = P[0] - B[0]
                PB[1] = P[1] - B[1]
                PB[2] = P[2] - B[2]

                overlap_ss = math.exp(
                    -alphaA * alphaB * AB2 * ooa) * math.sqrt(
                    math.pi * ooa) * math.pi * ooa * coeffA * coeffB

                (x, y, z) = os_recursion(PA, PB, alpha, AMa + 1, AMb + 1)

                # Atomic orbital indexing
                AOab = 0

                for ii in range(AMa + 1):
                    l1 = AMa - ii
                    for jj in range(ii + 1):
                        m1 = ii - jj
                        n1 = jj

                        for kk in range(AMb + 1):
                            l2 = AMb - kk
                            for ll in range(kk + 1):
                                m2 = kk - ll
                                n2 = ll

                                x00 = x[l1, l2]
                                x10 = x[l1+1, l2]

                                y00 = y[m1, m2]
                                y10 = y[m1+1, m2]

                                z00 = z[n1, n2]
                                z10 = z[n1+1, n2]

                                DAx = (x10 + x00 * A[0]) * y00 * z00 * overlap_ss
                                DAy = x00 * (y10 + y00 * A[1]) * z00 * overlap_ss
                                DAz = x00 * y00 * (z10 + z00 * A[2]) * overlap_ss

                                buffer[0, AOab] -= DAx
                                buffer[1, AOab] -= DAy
                                buffer[2, AOab] -= DAz

                                AOab += 1

        buffer.shape = (3, Sa.ncartesian, Sb.ncartesian)
        return buffer

    def compute(self):
        """Returns a NumPy array containing the integrals."""
        S = np.zeros((3, self.basisA.nao(), self.basisB.nao()))
        nsA = self.basisA.nshell()
        nsB = self.basisB.nshell()

        for i in range(nsA):
            ni = self.basisA.shell(i).ncartesian
            firsti = self.basisA.shell(i).function_index

            for j in range(nsB):
                nj = self.basisB.shell(j).ncartesian
                firstj = self.basisB.shell(j).function_index

                # compute the pair
                buffer = self.compute_shell_pair(i, j)

                # for each integral that we received store it appropriately
                for p in range(ni):
                    for q in range(nj):
                        S[0, firsti + p, firstj + q] += buffer[0, p, q]
                        S[1, firsti + p, firstj + q] += buffer[1, p, q]
                        S[2, firsti + p, firstj + q] += buffer[2, p, q]

        return S

if __name__ == '__main__':
    molecule = p4.geometry(config['DEFAULT']['molecule'])
    basis = p4.core.BasisSet.build(molecule, 'BASIS', config['DEFAULT']['basis'], puream=False)
    mints = p4.core.MintsHelper(basis)

    SCF_MAX_ITER = int(config['SCF']['max_iter'])
    ndocc = int(config['DEFAULT']['nalpha'])

    print("Psi4 Overlap:")
    Sp = mints.ao_overlap().to_array()
    print(Sp)
    print("Homework Overlap:")
    Sint = OverlapInt(molecule, basis, basis)
    S = Sint.compute()
    print(S-Sp)

    print("Psi4 Kinetic")
    print(mints.ao_kinetic().to_array())
    print("Homework Kinetic")
    Kint = KineticInt(molecule, basis, basis)
    K = Kint.compute()

    V = mints.ao_potential().to_array()
    I = mints.ao_eri().to_array()

    H = K + V

    A = np.matrix(np.linalg.inv(la.sqrtm(S)))

    # print("Psi4 Dipole")
    # print([x.to_array() for x in mints.ao_dipole()])
    # print("Homework Dipole")
    Dint = DipoleInt(molecule, basis, basis)
    Dipole = Dint.compute()
    # print(Dint.compute())

    # Construct initial density matrix
    Ft = A.dot(H).dot(A)
    e, C = np.linalg.eigh(Ft)
    C = A.dot(C)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)

    E = 0.0
    Eold = 0.0
    Dold = np.zeros_like(D)

    for iteration in range(1, SCF_MAX_ITER + 1):

        # Build the Fock matrix
        J = np.einsum('pqrs,rs->pq', I, D)
        K = np.einsum('prqs,rs->pq', I, D)
        F = H + J * 2 - K

        # Calculate SCF energy
        E_SCF = np.einsum('pq,pq->', F + H,
                          D) + molecule.nuclear_repulsion_energy()
        print('RHF iteration %3d: energy %20.14f  dE %1.5E' % (
        iteration, E_SCF, (E_SCF - Eold)))

        if (abs(E_SCF - Eold) < 1.e-10):
            break

        Eold = E_SCF
        Dold = D

        #     print(F.dot(D.dot(S))-S.dot(D.dot(F)))

        # Transform the Fock matrix
        Ft = A.dot(F).dot(A)

        # Diagonalize the Fock matrix
        e, C = np.linalg.eigh(Ft)
        #     print(e)

        # Construct new SCF eigenvector matrix
        C = A.dot(C)

        # Form the density matrix
        Cocc = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)

