import os
#from pytest_shutil import cmdline

from addons import *
from utils import *


tdir = 'Tutorials/03_Hartree-Fock'

def test_3a(workspace):
    tfile = '3a_restricted-hartree-fock'
    exe_scriptified_ipynb(workspace, tdir, tfile)

def test_3b(workspace):
    tfile = '3b_rhf-diis'
    exe_scriptified_ipynb(workspace, tdir, tfile)

def test_3c(workspace):
    tfile = '3c_unrestricted-hartree-fock'
    exe_scriptified_ipynb(workspace, tdir, tfile)

def test_df(workspace):
    tfile = 'density-fitting'
    exe_scriptified_ipynb(workspace, tdir, tfile)

def hide_test_3a(workspace):
    tfile = '3a_restricted-hartree-fock'
    script = os.getcwd() + '/../' + tdir + '/' + tfile + '.ipynb'
    path = workspace.workspace
    workspace.run('jupyter nbconvert --to script ' + script + ' --output-dir=' + path)
    workspace.run('python ' + path + '/' + tfile + '.py')

