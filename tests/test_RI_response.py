from addons import *
from utils import *


tdir = 'Response-Theory'


def test_beta(workspace):
    exe_py(workspace, tdir, 'Self-Consistent-Field/beta')


def test_CPHF(workspace):
    exe_py(workspace, tdir, 'Self-Consistent-Field/CPHF')


def test_helper_CPHF(workspace):
    exe_py(workspace, tdir, 'Self-Consistent-Field/helper_CPHF')


def test_TDHF(workspace):
    exe_py(workspace, tdir, 'Self-Consistent-Field/TDHF')


def test_polar_cc(workspace):
    exe_py(workspace, tdir, 'Coupled-Cluster/RHF/polar')


def test_optrot_cc(workspace):
    exe_py(workspace, tdir, 'Coupled-Cluster/RHF/optrot')
