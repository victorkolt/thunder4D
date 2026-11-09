import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # GUI tests: no window, no display needed
