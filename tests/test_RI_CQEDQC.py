from addons import *
from utils import *


tdir = 'Polaritonic-Quantum-Chemistry'


def test_CQED_RHF(workspace):
    exe_py(workspace, tdir, 'CQED_RHF')


def test_CS_CQED_CIS(workspace):
    exe_py(workspace, tdir, 'CS_CQED_CIS')

