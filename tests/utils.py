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
