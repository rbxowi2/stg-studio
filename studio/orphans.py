"""Find unattended training runs: processes left behind after the studio crashed or was closed.

They are only listed, never killed automatically. Killing would mean
guessing that a run is no longer wanted, and a healthy run must never be
killed by mistake.

`joblimit` covers runs started from now on (they are bound to the studio's
lifetime). This module covers runs that are already outside, including runs
started from the command line.

## Identifying the same process

`Runner.start()` writes the PID and the process creation time to
`_studio/pid.txt`. Comparing PIDs alone is not enough because PIDs are
reused; both must match (creation times within 2 s), otherwise the run is
treated as finished.

## Releasing

Only `stop.flag` is sent (the run finishes cleanly and frees GPU memory).
This module never force-kills; that decision is left to the user.
"""
import ctypes
import os
import sys
from ctypes import wintypes

PID_NAME = "pid.txt"

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform == "win32" else None


_STILL_ACTIVE = 259


def _created_at(pid):
    """Creation time (Unix seconds) of a running process. None if it has exited.

    Being able to open a handle does not mean the process is alive: an exited
    process that someone still holds a handle to still answers `OpenProcess`
    and `GetProcessTimes`. The exit code is checked as well.
    """
    if _k32 is None:
        return None
    h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        creation = wintypes.FILETIME()
        rest = (wintypes.FILETIME * 3)()
        if not _k32.GetProcessTimes(h, ctypes.byref(creation),
                                    ctypes.byref(rest[0]),
                                    ctypes.byref(rest[1]),
                                    ctypes.byref(rest[2])):
            return None
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return None
        if code.value != _STILL_ACTIVE:
            return None                            # already exited
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return ticks / 1e7 - 11644473600.0        # FILETIME counts from 1601
    finally:
        _k32.CloseHandle(h)


def write_pid(ctrl, pid):
    """Record a run as soon as it starts (called by `Runner.start()`). Failure does not affect training."""
    try:
        with open(os.path.join(ctrl, PID_NAME), "w", encoding="utf-8") as f:
            f.write(f"{pid} {_created_at(pid) or 0:.3f}\n")
    except OSError:
        pass


def clear_pid(ctrl):
    try:
        os.remove(os.path.join(ctrl, PID_NAME))
    except OSError:
        pass


def _read_pid(path):
    try:
        parts = open(path, encoding="utf-8").read().split()
        return int(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
    except (OSError, ValueError, IndexError):
        return None, None


def scan(root, ctrl_dir="_studio", skip=()):
    """Training runs under this shooting root that are alive but unattended.

    `skip` holds the PIDs this studio is running itself; those are not orphans.
    Returns [{"pid", "name", "ctrl", "started"}] sorted by start time.
    """
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        ctrl = os.path.join(root, name, ctrl_dir)
        pid, born = _read_pid(os.path.join(ctrl, PID_NAME))
        if pid is None or pid in skip:
            continue
        now = _created_at(pid)
        if now is None:
            continue                       # the process ended long ago; the file is stale
        if born and abs(now - born) > 2.0:
            continue                       # the PID was reused by another process
        out.append({"pid": pid, "name": name, "ctrl": ctrl, "started": now})
    return sorted(out, key=lambda d: d["started"])


def release(entry):
    """Ask the run to finish by itself (send stop.flag). True if the flag was written.

    This neither waits nor force-kills: the run may be writing a file of
    hundreds of megabytes, and killing it mid-write leaves a corrupt file.
    It stays listed until it disappears by itself.
    """
    try:
        with open(os.path.join(entry["ctrl"], "stop.flag"), "w") as f:
            f.write("")
        return True
    except OSError:
        return False
