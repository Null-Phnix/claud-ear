#!/usr/bin/env python3
"""macOS idle/battery detection for throttling the agent."""
import subprocess
import time


def get_idle_seconds() -> float:
    """Return seconds since last keyboard/mouse activity using ioreg."""
    try:
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "HIDIdleTime" in line:
                # Line looks like: | "HIDIdleTime" = 1234567890
                parts = line.split("=")
                if len(parts) >= 2:
                    ns = int(parts[-1].strip())
                    return ns / 1e9  # nanoseconds → seconds
    except Exception:
        pass
    return 0.0


def is_user_active() -> bool:
    """Return True if the user has been active within the last 60 seconds."""
    return get_idle_seconds() < 60


def sleep_between_songs():
    """
    Throttle between song analyses.
    If user is active: sleep 30s to save battery.
    If idle: sleep 5s to run fast.
    """
    if is_user_active():
        time.sleep(30)
    else:
        time.sleep(5)


if __name__ == "__main__":
    idle = get_idle_seconds()
    active = is_user_active()
    print(f"Idle time: {idle:.1f}s")
    print(f"User active: {active}")
