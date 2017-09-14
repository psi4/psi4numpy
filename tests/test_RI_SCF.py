import os

from addons import *
from utils import *


tdir = 'Self-Consistent-Field'

def test_CPHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/CPHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_RHF_DIIS(workspace):
    script = os.getcwd() + '/../' + tdir + '/RHF_DIIS.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_RHF_libJK(workspace):
    script = os.getcwd() + '/../' + tdir + '/RHF_libJK.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_RHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/RHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_ROHF_libJK(workspace):
    script = os.getcwd() + '/../' + tdir + '/ROHF_libJK.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SORHF_iterative(workspace):
    script = os.getcwd() + '/../' + tdir + '/SORHF_iterative.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SORHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/SORHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

@using_scipy
def test_SOROHF_iterative(workspace):
    script = os.getcwd() + '/../' + tdir + '/SOROHF_iterative.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SOROHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/SOROHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SOUHF_iterative(workspace):
    script = os.getcwd() + '/../' + tdir + '/SOUHF_iterative.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SOUHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/SOUHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_TDHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/TDHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_UHF_libJK(workspace):
    script = os.getcwd() + '/../' + tdir + '/UHF_libJK.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)
