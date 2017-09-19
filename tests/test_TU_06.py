from addons import *
from utils import *


tdir = 'Tutorials/06_Molecular_Properties'


def test_CPSCF(workspace):
    exe_scriptified_ipynb(workspace, tdir, 'CP-SCF')
