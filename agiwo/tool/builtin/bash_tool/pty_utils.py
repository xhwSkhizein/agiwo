"""Shared PTY utilities for bash tool modules."""

import fcntl
import struct
import termios


def set_pty_size(fd: int, cols: int, rows: int) -> None:
    """Set the terminal size on a PTY file descriptor."""
    if cols <= 0 or rows <= 0:
        return
    size = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    except OSError:
        return
