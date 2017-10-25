from addons import *
from utils import *


tdir = 'Self-Consistent-Field'


def test_CPHF(workspace):
    exe_py(workspace, tdir, 'CPHF')


def test_RHF_DIIS(workspace):
    exe_py(workspace, tdir, 'RHF_DIIS')


def test_RHF_libJK(workspace):
    exe_py(workspace, tdir, 'RHF_libJK')


def test_RHF(workspace):
    exe_py(workspace, tdir, 'RHF')

@using_psi4_sadpy
def test_RHF_libSADGuess(workspace):
    exe_py(workspace, tdir, 'RHF_libSADGuess')


@using_psi4_efpmints
@using_pylibefp
def test_RHF_EFP(workspace):
    exe_py(workspace, tdir, 'RHF_EFP')


def test_ROHF_libJK(workspace):
    exe_py(workspace, tdir, 'ROHF_libJK')


def test_SORHF_iterative(workspace):
    exe_py(workspace, tdir, 'SORHF_iterative')


def test_SORHF(workspace):
    exe_py(workspace, tdir, 'SORHF')


@using_scipy
def test_SOROHF_iterative(workspace):
    exe_py(workspace, tdir, 'SOROHF_iterative')


def test_SOROHF(workspace):
    exe_py(workspace, tdir, 'SOROHF')


def test_SOUHF_iterative(workspace):
    exe_py(workspace, tdir, 'SOUHF_iterative')


def test_SOUHF(workspace):
    exe_py(workspace, tdir, 'SOUHF')


def test_TDHF(workspace):
    exe_py(workspace, tdir, 'TDHF')


def test_UHF_libJK(workspace):
    exe_py(workspace, tdir, 'UHF_libJK')
