#!/usr/bin/env python3
"""Render the AppImage PNG icon from the SVG source."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: make_icon.py <input.svg> <output.png> [size]", file=sys.stderr)
        return 2
    source, destination = argv[1], argv[2]
    size = int(argv[3]) if len(argv) > 3 else 256

    app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(source)
    if not renderer.isValid():
        print(f"could not read SVG: {source}", file=sys.stderr)
        return 1

    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    if not image.save(destination, "PNG"):
        print(f"could not write PNG: {destination}", file=sys.stderr)
        return 1
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
