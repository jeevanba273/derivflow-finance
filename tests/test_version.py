"""Version is sourced from setuptools_scm; the two reads must never drift."""

import derivflow
import derivflow._version as scm_version


def test_version_single_source():
    assert derivflow.__version__ == scm_version.__version__


def test_version_not_hardcoded_placeholder():
    # The old hardcoded "1.0.0" must not reappear.
    assert derivflow.__version__ != "1.0.0"


def test_package_info_reports_runtime_version():
    info = derivflow.get_package_info()
    assert info['version'] == derivflow.__version__
    # The removed false claim must stay removed.
    assert all('Lookback' not in f for f in info['features'])
