"""
utils.py — shared spinner + timer for learning scripts
=======================================================
Used by all part*.py files. Not part of the main system.
"""

import sys
import time
import threading
from contextlib import contextmanager


# ── Spinner frames ────────────────────────────────────────────────────────────

FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """
    Animated spinner with elapsed time shown in the terminal.

    Usage:
        with Spinner("Calling LLM"):
            result = llm.invoke(...)
    """

    def __init__(self, label: str = "Thinking"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._start_time = 0.0

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            frame = FRAMES[i % len(FRAMES)]
            sys.stdout.write(f"\r  {frame}  {self.label}  {elapsed:.1f}s ")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def start(self):
        self._start_time = time.time()
        self._thread.start()

    def stop(self, success: bool = True):
        self._stop_event.set()
        self._thread.join()
        elapsed = time.time() - self._start_time
        icon = "✓" if success else "✗"
        sys.stdout.write(f"\r  {icon}  {self.label}  {elapsed:.2f}s\n")
        sys.stdout.flush()
        return elapsed


@contextmanager
def spinner(label: str = "Thinking"):
    """Context manager — wraps any blocking call with a spinner."""
    s = Spinner(label)
    s.start()
    try:
        yield s
        s.stop(success=True)
    except Exception:
        s.stop(success=False)
        raise

# ── Elapsed timer (no spinner — for streaming where text is already flowing) ──

class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def print_elapsed(self, label: str = "Done"):
        print(f"\n  ✓ {label}  {self.elapsed():.2f}s")