from addons import *
from utils import *


tdir = 'Tutorials/03_Hartree-Fock'


def test_3a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '3a_restricted-hartree-fock')


def test_3b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '3b_rhf-diis')


def test_3c(workspace):
    exe_scriptified_ipynb(workspace, tdir, '3c_unrestricted-hartree-fock')


def test_df(workspace):
    exe_scriptified_ipynb(workspace, tdir, 'density-fitting')
