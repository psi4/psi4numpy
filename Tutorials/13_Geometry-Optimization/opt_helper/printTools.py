from __future__ import print_function

def printMat(M, Ncol=7, title=None):
    if title:
        print(title + '\n')
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            print(" %10.6f" % M[row, col])
            if tab == Ncol and col != (M.shape[1] - 1):
                print("\n")
                tab = 0
        print("\n")
    return


def printMatString(M, Ncol=7, title=None):
    if title:
        print(title + '\n')
    s = ''
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            s += " %10.6f" % M[row, col]
            if tab == Ncol and col != (M.shape[1] - 1):
                s += '\n'
                tab = 0
        s += '\n'
    return s


def printArray(M, Ncol=7, title=None):
    if title:
        print(title + '\n')
    tab = 0
    for col, entry in enumerate(M):
        tab += 1
        print(" %10.6f" % M[col])
        if tab == Ncol and col != (len(M) - 1):
            print("\n")
            tab = 0
    print("\n")
    return


def printArrayString(M, Ncol=7, title=None):
    if title:
        print(title + '\n')
    tab = 0
    s = ''
    for i, entry in enumerate(M):
        tab += 1
        s += " %10.6f" % entry
        if tab == Ncol and i != (len(M) - 1):
            s += '\n'
            tab = 0
    s += '\n'
    return s


def printGeomGrad(geom, grad):
    print("\tGeometry and Gradient\n")
    Natom = geom.shape[0]

    for i in range(Natom):
        print("\t%20.10f%20.10f%20.10f\n" % (geom[i, 0], geom[i, 1], geom[i, 2]))
    print("\n")
    for i in range(Natom):
        print("\t%20.10f%20.10f%20.10f\n" % (grad[3 * i + 0], grad[3 * i + 1],
                                                 grad[3 * i + 2]))
