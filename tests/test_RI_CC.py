from addons import *
from utils import *


tdir = 'Coupled-Cluster'


def test_CCSD_DIIS(workspace):
    exe_py(workspace, tdir+"/Spin_Orbitals/CCSD", 'CCSD_DIIS')


def test_CCSD(workspace):
    exe_py(workspace, tdir+"/Spin_Orbitals/CCSD", 'CCSD')


def test_CCSD_T(workspace):
    exe_py(workspace, tdir+"/Spin_Orbitals/CCSD", 'CCSD_T')


def test_EOM_CCSD(workspace):
    exe_py(workspace, tdir+'/RHF','EOM_CCSD')


#def test_TD_CCSD(workspace):
#    exe_py(workspace, tdir, 'TD-CCSD')
