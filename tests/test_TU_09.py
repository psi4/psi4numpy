from addons import *
from utils import *


tdir = 'Tutorials/09_Configuration_Interaction'


@using_numpy_113
def test_9a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '9a_cis')
