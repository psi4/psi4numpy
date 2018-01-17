import pytest
from addons import *
from utils import *


tdir = 'Symmetry-Adapted-Perturbation-Theory'


def test_SAPT0ao(workspace):
    exe_py(workspace, tdir, 'SAPT0ao')


def test_SAPT0(workspace):
    exe_py(workspace, tdir, 'SAPT0')


@pytest.mark.long
def test_SAPT0_ROHF(workspace):
    exe_py(workspace, tdir, 'SAPT0_ROHF')


def test_SAPT0_no_S2(workspace):
    exe_py(workspace, tdir, 'SAPT0_no_S2')
