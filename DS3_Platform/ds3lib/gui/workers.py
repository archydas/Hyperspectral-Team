from __future__ import annotations
from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    """Run a plain function off the UI thread; report the result or the error text."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:                      # shown to the user, never swallowed
            self.failed.emit(f"{type(e).__name__}: {e}")
