"""Switch USB serial ports to low-latency mode (``setserial <port> low_latency``) without changing anything else.

USB serial adapters can hold received bytes back for up to 16 ms (FTDI latency timer) before handing them to Linux;
in low-latency mode they are passed on at once. Only the ASYNC_LOW_LATENCY flag is set (TIOCSSERIAL): baud rate and
other line settings are not touched, so it is safe on a port another program (the IMU driver) is using.

    python3 -m balance_bot_policy.serial_low_latency                 # every /dev/ttyUSB* and /dev/ttyACM*
    python3 -m balance_bot_policy.serial_low_latency /dev/ttyUSB1    # only these

The setting lasts until the device is unplugged. Ports that do not support it (most ttyACM) are reported and skipped.
"""

import array
import fcntl
import glob
import os
import sys
import termios

ASYNC_LOW_LATENCY = 0x2000
TIOCGSERIAL = getattr(termios, "TIOCGSERIAL", 0x541E)
TIOCSSERIAL = getattr(termios, "TIOCSSERIAL", 0x541F)


def set_low_latency_fd(fd: int) -> bool:
    """Set ASYNC_LOW_LATENCY on an open tty. Returns True if it is set afterwards."""
    buf = array.array("i", [0] * 32)  # struct serial_struct, flags at index 4
    fcntl.ioctl(fd, TIOCGSERIAL, buf)
    if buf[4] & ASYNC_LOW_LATENCY:
        return True
    buf[4] |= ASYNC_LOW_LATENCY
    fcntl.ioctl(fd, TIOCSSERIAL, buf)
    fcntl.ioctl(fd, TIOCGSERIAL, buf)
    return bool(buf[4] & ASYNC_LOW_LATENCY)


def set_low_latency(port: str) -> str:
    """Set it on a port by name (opened non-blocking, line settings untouched). Returns a status message."""
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as e:
        return f"{port}: cannot open ({e.strerror})"
    try:
        ok = set_low_latency_fd(fd)
        return f"{port}: low latency {'on' if ok else 'NOT supported'}"
    except OSError as e:
        return f"{port}: low latency not supported ({e.strerror})"
    finally:
        os.close(fd)


def main():
    ports = sys.argv[1:] or sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        print("serial low latency: no /dev/ttyUSB* or /dev/ttyACM* ports")
    for port in ports:
        print("serial low latency:", set_low_latency(port))


if __name__ == "__main__":
    main()
