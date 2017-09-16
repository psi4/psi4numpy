from addons import *
from utils import *


tdir = 'Tutorials/01_Psi4NumPy-Basics'


def test_1a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1a_Getting-Started')


@using_matplotlib
def test_1b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1b_molecule')


def test_1c(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1c_psi4-numpy-datasharing')


def test_1d(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1d_wavefunction')


def test_1e(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1e_mints-helper')


def test_1f(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1f_tensor-manipulation')


def test_1g(workspace):
    exe_scriptified_ipynb(workspace, tdir, '1g_basis-sets')
