from addons import *
from utils import *


tdir = 'Tutorials/04_Density_Functional_Theory'


@using_matplotlib
@using_psi4_libxc
def test_4a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '4a_DFT_Grid')


@using_psi4_libxc
def test_4b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '4b_LDA_kernel')


@using_psi4_libxc
def test_4d(workspace):
    copy_helpers(workspace, tdir, files=['ks_helper.py'])
    exe_scriptified_ipynb(workspace, tdir, '4d_VV10')


@using_matplotlib
@using_psi4_libxc
def test_4e(workspace):
    copy_helpers(workspace, tdir, files=['ks_helper.py'])
    exe_scriptified_ipynb(workspace, tdir, '4e_GRAC')
