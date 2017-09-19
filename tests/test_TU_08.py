import pytest
from addons import *
from utils import *


tdir = 'Tutorials/08_CEPA0_and_CCD'


@using_numpy_113
def test_8a(workspace):
    exe_scriptified_ipynb(workspace, tdir, '8a_Intro_to_spin_orbital_postHF')


@pytest.mark.long
@using_numpy_113
def test_8b(workspace):
    exe_scriptified_ipynb(workspace, tdir, '8b_CEPA0_and_CCD')
