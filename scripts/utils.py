#!/usr/bin/env python3


import os
import json
import math
import datetime as dt
import sys
import subprocess
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


_mute_level = 0  # severity floor read by `custom_print`; rebound only by `muted`


def custom_print(msg: object = "", level: str = "INFO", with_time: bool = False):
    '''
    Print one message, tagged by its severity and optionally time-stamped.

    Messages below the current severity floor are dropped (raise it with `muted`):
    0 prints everything, 1 mutes INFO, 2 also mutes WARNING, 3 mutes all.

    Note: the floor is compared against the position of `level` in `tags`, 
    so a new level must be inserted in rank order.
    '''
    tags = {  # also serve as the severity ordering
        "INFO"   : "", 
        "WARNING": "Warning: ", 
        "ERROR"  : "Error: "
    }
    level = level.upper()  # case-insensitive

    if list(tags.keys()).index(level) < _mute_level:
        return

    body = str(msg).strip("\n")  # msg as a string, with all leading and trailing newline characters removed

    if body:  # keep pure separators (e.g. "" or "\n") unadorned
        lead, trail = str(msg).split(body)
        stamp = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] " if with_time else ""
        body = f"{lead}{stamp}{tags[level]}{body}{trail}"

    print(body, flush=True)


@contextmanager
def mute_print(level: int = 2):
    '''
    Control the severity floor of `custom_print` 
    for the duration of the `with` block.
    '''
    global _mute_level

    assert isinstance(level, int), f"\n'level' should be an int, got {type(level).__name__}\n"

    prev = _mute_level
    _mute_level = level

    try:
        yield

    finally:
        _mute_level = prev


def to_json_compatible(data, nan_to_none: bool = True):
    '''
    Recursively convert data into built-in Python types accepted by `json.dump`
    (the input is assumed to be a finite tree).

    If `nan_to_none` is True, 
    `float('nan')`, `float('inf')`, and `float('-inf')` are convert to `None`; 
    otherwise, by default, `json.dumps` maps them to NaN, Infinity, and -Infinity 
    (or you can disable this non-standard behavior by setting `allow_nan=False`).
    '''
    ## leaf types that are already JSON-safe -------------------------------

    if (data is None) or (data is pd.NA) or (data is pd.NaT):
        return None

    if isinstance(data, (str, bool)):  # before int: bool is an int subclass
        return data

    if isinstance(data, float):
        if math.isfinite(data):
            return data
        return None if nan_to_none else data

    if isinstance(data, int):
        return data

    ## numpy scalars -------------------------------------------------------

    if isinstance(data, np.datetime64):
        return to_json_compatible(pd.Timestamp(data), nan_to_none)  # convert to dt.datetime with uniform unit 

    if isinstance(data, np.timedelta64):
        return to_json_compatible(pd.Timedelta(data), nan_to_none)

    if isinstance(data, np.generic):  # np.int64, np.float32, np.bool_, np.str_, ...
        return to_json_compatible(data.item(), nan_to_none)

    ## containers ----------------------------------------------------------
    
    if isinstance(data, dict):
        def _as_key(key):
            k = to_json_compatible(key, nan_to_none)
            if k is None or isinstance(k, (str, bool, int)) or (isinstance(k, float) and math.isfinite(k)):
                return k
            return str(k)

        return { _as_key(k): to_json_compatible(v, nan_to_none) for k, v in data.items() }

    if isinstance(data, np.ndarray):
        if data.dtype.kind in "mM":  # datetime64 / timedelta64: .tolist() drops to ints
            return (
                to_json_compatible(data[()], nan_to_none) if data.ndim == 0 
                else [ to_json_compatible(item, nan_to_none) for item in data ]
            )
        return to_json_compatible(data.tolist(), nan_to_none)  # object arrays still hold Python objects

    if isinstance(data, (list, tuple)):
        return [ to_json_compatible(item, nan_to_none) for item in data ]

    if isinstance(data, (set, frozenset)):
        try:
            items = sorted(data)
        except TypeError:  # mutually incomparable element types
            items = list(data)
        return [ to_json_compatible(item, nan_to_none) for item in items ]

    if isinstance(data, pd.DataFrame):
        return to_json_compatible(data.to_dict(), nan_to_none)  # {column: {index: value}}

    if isinstance(data, pd.Series):
        return to_json_compatible(data.to_dict(), nan_to_none)  # {index: value}

    if isinstance(data, pd.Index):
        return to_json_compatible(data.tolist(), nan_to_none)

    ## remaining stdlib leaf types ----------------------------------------
    
    if isinstance(data, (dt.datetime, dt.date, dt.time)):  # covers pd.Timestamp
        return data.isoformat()

    if isinstance(data, dt.timedelta):  # covers pd.Timedelta
        return data.total_seconds()

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, (bytes, bytearray)):
        return bytes(data).decode("utf-8", errors="replace")  # lossy for non-UTF-8 payloads

    if isinstance(data, Decimal):
        return to_json_compatible(float(data), nan_to_none)

    if isinstance(data, complex):
        return [to_json_compatible(data.real, nan_to_none), to_json_compatible(data.imag, nan_to_none)]

    return data


@contextmanager
def tee_output(log_path: Path):
    '''
    Duplicate stdout/stderr to both the terminal and `log_path`.

    Redirection happens at the file-descriptor level, 
    so it also captures output from C extensions and joblib/loky workers, 
    not just `print`.
    '''
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()  # clears the internal memory 
    sys.stderr.flush()

    proc = subprocess.Popen(["tee", str(log_path)], stdin=subprocess.PIPE)
    saved_fds = (os.dup(1), os.dup(2))  # file descriptors
    os.dup2(proc.stdin.fileno(), 1)
    os.dup2(proc.stdin.fileno(), 2)
    was_line_buffered = sys.stdout.line_buffering
    sys.stdout.reconfigure(line_buffering=True)  # keep ordering vs. stderr / workers
    
    try:
        yield

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.reconfigure(line_buffering=was_line_buffered)
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in saved_fds:
            os.close(fd)
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # the resource trackers of loky / multiprocessing hold the pipe until this process exits
            pass


