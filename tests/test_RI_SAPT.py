import os

from addons import *
from utils import *


tdir = 'Symmetry-Adapted-Perturbation-Theory'

def test_SAPT0ao(workspace):
    script = os.getcwd() + '/../' + tdir + '/SAPT0ao.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SAPT0(workspace):
    script = os.getcwd() + '/../' + tdir + '/SAPT0.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

def test_SAPT0_ROHF(workspace):
    script = os.getcwd() + '/../' + tdir + '/SAPT0_ROHF.py'
    with uplusx(script) as exescript:
        workspace.run(exescript)

