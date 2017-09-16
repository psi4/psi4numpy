from addons import *
from utils import *


tdir = 'Tutorials/05_Moller-Plesset'


def test_5a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '5a_conventional-mp2')


def test_5b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '5b_density-fitted-mp2')
