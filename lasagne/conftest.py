ignore_test_paths = [
    "*/layers/corrmm.py",
    "*/layers/cuda_convnet.py",
    "*/layers/dnn.py",
    ]


def pytest_ignore_collect(path, config):
    """Ignore paths that would otherwise be collceted by the doctest
    plugin and lead to ImportError due to missing dependencies.
    """
    return any(path.fnmatch(ignore) for ignore in ignore_test_paths)
