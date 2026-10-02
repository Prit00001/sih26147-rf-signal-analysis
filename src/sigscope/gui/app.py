"""GUI entry point.

Covers: FR-14, NFR-03.
Run: python -m sigscope.gui.app
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from sigscope.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
