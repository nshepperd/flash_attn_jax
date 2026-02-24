"""Test configuration to enter debugging in vscode when tests fail."""

import os
os.environ['XLA_CLIENT_MEM_FRACTION'] = '0.25'
import pytest
import sys
import threading
import jax

jax.config.update("jax_default_matmul_precision", "highest")

def maybe_debugpy_postmortem(excinfo):
    """Make the debugpy debugger enter and stop at a raised exception.

    excinfo: A (type(e), e, e.__traceback__) tuple. See sys.exc_info()
    """
    try:
        import debugpy
        import pydevd  # type: ignore
    except ImportError:
        # If pydevd isn't available, no debugger attached; do nothing.
        return

    if not debugpy.is_client_connected():
        return

    py_db = pydevd.get_global_debugger()
    thread = threading.current_thread()
    additional_info = py_db.set_additional_thread_info(thread)
    additional_info.is_tracing += 1
    try:
        py_db.stop_on_unhandled_exception(py_db, thread, additional_info, excinfo)
    finally:
        additional_info.is_tracing -= 1

if sys.gettrace() is not None:
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo):
        print(f"pytest_exception_interact called with call: {call}")
        if call.when == 'call' and call.excinfo and call.excinfo._excinfo:
            maybe_debugpy_postmortem(call.excinfo._excinfo)