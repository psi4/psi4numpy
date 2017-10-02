from addons import *
from utils import *


tdir = 'Tutorials/12_Ewald_Summation'


@using_matplotlib
def test_12a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '12a_ewald')

