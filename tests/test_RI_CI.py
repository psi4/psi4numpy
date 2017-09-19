from addons import *
from utils import *


tdir = 'Configuration-Interaction'


def test_CI_DL(workspace):
    exe_py(workspace, tdir, 'CI_DL')


@using_scipy
def test_CISD(workspace):
    exe_py(workspace, tdir, 'CISD')


def test_CIS(workspace):
    exe_py(workspace, tdir, 'CIS')


@using_scipy
def test_FCI(workspace):
    exe_py(workspace, tdir, 'FCI')
