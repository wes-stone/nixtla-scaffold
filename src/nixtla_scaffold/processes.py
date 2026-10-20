"""Bounded script execution with owned descendant cleanup on timeout."""
from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import psutil


class ProcessCancelled(RuntimeError):
    pass


def run_script(command: list[str], *, cwd: Path, timeout: float,
               cancelled: Callable[[], bool] | None = None,
               input_text: str | None = None, output_limit: int = 64 * 1024) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr, tempfile.TemporaryFile() as stdin:
        if input_text is not None:
            stdin.write(input_text.encode("utf-8"))
            stdin.seek(0)
        process = subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, shell=False)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    raise ProcessCancelled("Owned script was cancelled.")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(.1, max(.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    continue
        except (subprocess.TimeoutExpired, ProcessCancelled, KeyboardInterrupt):
            try:
                owner = psutil.Process(process.pid)
                descendants = owner.children(recursive=True)
                for child in reversed(descendants):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                owner.kill()
                _, survivors = psutil.wait_procs([*descendants, owner], timeout=5)
                if survivors:
                    raise RuntimeError("Timed-out script cleanup is incomplete; recover the owned job before retrying.")
            except psutil.NoSuchProcess:
                pass
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            raise
        def tail(stream):
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - output_limit))
            return stream.read().decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(command, process.returncode, tail(stdout), tail(stderr))
