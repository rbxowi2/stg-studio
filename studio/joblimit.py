"""Bind child processes to a job object: when the studio exits, Windows ends them too.

Without this, child processes have no lifetime relationship with the studio:
if the studio crashes, a training process keeps running and keeps holding
system and GPU memory, with no window left to stop it.

    from joblimit import adopt
    p = subprocess.Popen(...)
    adopt(p.pid)          # from now on it ends whenever the studio ends

The criterion is only "does the studio process still exist", answered by the
operating system. No heartbeat or timeout is used: training legitimately goes
quiet for long periods (saving, rendering a video, paused, finished and
waiting), and a time-based rule would eventually kill a healthy run.
A job object does nothing while the studio is alive; the kernel acts only
when the last handle to the job closes, i.e. when this process ends for any
reason (normal exit, crash, killed from Task Manager).

The handle must live until the process ends: closing it is the same as
declaring the studio dead, and the children would be killed immediately.
It is therefore kept in the module-level `_JOB` and never closed.

Windows only. On other platforms `adopt()` returns False and changes nothing.
"""
import ctypes
import sys
from ctypes import wintypes

_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

_JOB = None                  # the job handle (one per process)
_TRIED = False               # creation attempted (do not retry after a failure)

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
else:                                                     # pragma: no cover
    _k32 = None


class _BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IO(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64)]


class _EXT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _job():
    """Create (or return) the kill-on-close job. Returns None if it cannot be created."""
    global _JOB, _TRIED
    if _JOB is not None or _TRIED:
        return _JOB
    _TRIED = True
    if _k32 is None:
        return None
    h = _k32.CreateJobObjectW(None, None)
    if not h:
        return None
    info = _EXT()
    info.BasicLimitInformation.LimitFlags = _JOB_LIMIT_KILL_ON_JOB_CLOSE
    if not _k32.SetInformationJobObject(
            h, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info)):
        _k32.CloseHandle(h)
        return None
    _JOB = h
    return _JOB


def adopt(pid):
    """Bind a process (and every descendant it starts) to this process's lifetime.

    Returns whether it succeeded. Failure returns False instead of raising:
    the worst case is the unbound behaviour (a possible orphan), which must
    not prevent training from starting.
    """
    h = _job()
    if not h:
        return False
    ph = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not ph:
        return False
    try:
        return bool(_k32.AssignProcessToJobObject(h, ph))
    finally:
        _k32.CloseHandle(ph)
