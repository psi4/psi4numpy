import os
from contextlib import contextmanager


@contextmanager
def uplusx(fd):
    """Context Manager to turn ``fd`` executable, then reset it to rw-rw-r--"""
    try:
        os.chmod(fd, 0o744)
        yield fd
    finally:
        os.chmod(fd, 0o664)


def exe_py(workspace, tdir, py):
    script = os.getcwd() + '/../' + tdir + '/' + py + '.py'
    workspace.run('python ' + script)


def exe_scriptified_ipynb(workspace, tdir, ipynb):
    script = os.getcwd() + '/../' + tdir + '/' + ipynb + '.ipynb'
    path = workspace.workspace
    workspace.run('jupyter nbconvert --to script ' + script + ' --output-dir=' + path)
    workspace.run('python ' + path + '/' + ipynb + '.py')

