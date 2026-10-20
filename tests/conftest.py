import pytest
import os


@pytest.fixture(autouse=True)
def isolated_evidence_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("NIXTLA_EVIDENCE_DIRECTORY", str(tmp_path / ".nixtla-evidence"))


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("NIXTLA_FAIL_ON_SKIP") == "1":
        reporter = session.config.pluginmanager.getplugin("terminalreporter")
        if reporter and reporter.stats.get("skipped"):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
