import pytest


@pytest.fixture(scope="session", autouse=True)
def set_up_overall(request):
    try:
        import pytest_shutil
    except ImportError:
        raise Exception("Please ``pip install pytest-shutil`` to run the tests.")


@pytest.fixture(scope="function", autouse=True)
def set_up():
    pass


def tear_down():
    pass
