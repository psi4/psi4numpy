import os
import re
import shutil
import tempfile
from contextlib import contextmanager


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextmanager
def uplusx(fd):
    """Context Manager to turn ``fd`` executable, then reset it to rw-rw-r--"""
    try:
        os.chmod(fd, 0o744)
        yield fd
    finally:
        os.chmod(fd, 0o664)


def exe_py(workspace, tdir, py):
    script = base_dir + '/' + tdir + '/' + py + '.py'
    workspace.run('python ' + script)


def exe_scriptified_ipynb(workspace, tdir, ipynb):
    script = base_dir + '/' + tdir + '/' + ipynb + '.ipynb'
    path = workspace.workspace
    workspace.run('jupyter nbconvert --to script ' + script + ' --output-dir=' + path)
    script_py = path + '/' + ipynb + '.py'
    sed_inplace(script_py,
                r"""get_ipython\(\).magic\(u?'matplotlib inline'\)""",
                r"""# <<<  Jupyter magic  >>>  get_ipython().magic('matplotlib inline')\nimport matplotlib as mpl; mpl.use('Agg')""")
    sed_inplace(script_py,
                r"""get_ipython\(\).magic\(u?'matplotlib notebook'\)""",
                r"""# <<<  Jupyter magic  >>>  get_ipython().magic('matplotlib notebook')\nimport matplotlib as mpl; mpl.use('Agg')""")
    sed_inplace(script_py,
                r"""get_ipython\(\).magic\(u?['"]timeit """,
                r"""# <<<  Jupyter magic  >>>""")
    sed_inplace(script_py,
                r"""get_ipython\(\).run_line_magic\(u?'matplotlib', 'inline'\)""",
                r"""# <<<  Jupyter magic  >>>  get_ipython().run_line_magic('matplotlib', 'inline')\nimport matplotlib as mpl; mpl.use('Agg')""")
    sed_inplace(script_py,
                r"""get_ipython\(\).run_line_magic\(u?'matplotlib', 'notebook'\)""",
                r"""# <<<  Jupyter magic  >>>  get_ipython().run_line_magic('matplotlib', 'notebook')\nimport matplotlib as mpl; mpl.use('Agg')""")
    sed_inplace(script_py,
                r"""get_ipython\(\).run_line_magic\(u?'timeit'""",
                r"""# <<<  Jupyter magic  >>> get_ipython().run_line_magic('timeit'""")
    # Allow use of __file__ for original notebook path.
    sed_inplace(script_py,
                r"""__file__""",
                """'{}'""".format(os.path.abspath(script)))
    workspace.run('python ' + script_py)


def copy_helpers(workspace, tdir, files):
    from_dir = base_dir + '/' + tdir + '/'
    for fl in files:
        shutil.copy(from_dir + fl, workspace.workspace)


# from https://stackoverflow.com/a/31499114
def sed_inplace(filename, pattern, repl):
    """Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.

    Examples
    --------
    sed_inplace('/etc/apt/sources.list', r'^\# deb', 'deb')

    """
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)
