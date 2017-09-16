from addons import *
from utils import *


tdir = 'Tutorials/11_Integrals'


def test_11a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '11a_1e_Integrals')


def test_11b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '11b_Appendix')
