"""tenplaces.cores.no_power_throttling: the process opts out of Windows power throttling (EcoQoS); a no-op elsewhere."""
import ctypes
import sys

import pytest

from tenplaces.cores import no_power_throttling


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling")
def test_opt_out_is_set_for_this_process():
    from ctypes import wintypes

    assert no_power_throttling() is True

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0, 0)
    assert k32.GetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))
    assert (s.ControlMask & 1, s.StateMask & 1) == (1, 0)  # execution speed controlled, not throttled


def test_elsewhere_it_does_nothing(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert no_power_throttling() is False
