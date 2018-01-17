from addons import *
from utils import *


tdir = 'Tutorials/06_Molecular_Properties'


def test_6a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '6a_CP-SCF')


def test_6b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '6b_first_hyperpolarizability')
