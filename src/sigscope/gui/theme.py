"""Dark QSS theme for the sigscope GUI.

Covers: FR-14 (GUI), NFR-03 (usable without coding).
"""

from __future__ import annotations

DARK_QSS = """
QWidget {
    background-color: #1e1f22;
    color: #d4d4d4;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #1e1f22;
}
QTabWidget::pane {
    border: 1px solid #3a3d41;
    background-color: #1e1f22;
}
QTabBar::tab {
    background-color: #2b2d30;
    color: #b3b3b3;
    padding: 6px 12px;
    border: 1px solid #3a3d41;
}
QTabBar::tab:selected {
    background-color: #3574f0;
    color: #ffffff;
}
QPushButton {
    background-color: #3574f0;
    color: #ffffff;
    border-radius: 4px;
    padding: 6px 12px;
    border: none;
}
QPushButton:hover {
    background-color: #4a84f5;
}
QPushButton:disabled {
    background-color: #4b4f56;
    color: #8a8a8a;
}
QTableWidget, QListWidget, QPlainTextEdit, QTextEdit {
    background-color: #2b2d30;
    color: #d4d4d4;
    gridline-color: #3a3d41;
    border: 1px solid #3a3d41;
}
QHeaderView::section {
    background-color: #2b2d30;
    color: #b3b3b3;
    padding: 4px;
    border: 1px solid #3a3d41;
}
QProgressBar {
    background-color: #2b2d30;
    border: 1px solid #3a3d41;
    border-radius: 3px;
    text-align: center;
    color: #d4d4d4;
}
QProgressBar::chunk {
    background-color: #3574f0;
}
QLabel {
    color: #d4d4d4;
}
"""
