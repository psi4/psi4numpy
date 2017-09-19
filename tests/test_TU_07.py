from addons import *
from utils import *


tdir = 'Tutorials/07_Symmetry_Adapted_Perturbation_Theory'


def test_7a(workspace):
    copy_helpers(workspace, tdir, files=['helper_SAPT.py'])
    exe_scriptified_ipynb(workspace, tdir, '7a_sapt0_mo')


def test_7b(workspace):
    copy_helpers(workspace, tdir, files=['helper_SAPT.py'])
    exe_scriptified_ipynb(workspace, tdir, '7b_sapt0_ao')
