"""Programmatic test PDF creation using PyMuPDF.

Creates simple PDF files with known geometry for parser testing.
"""

from __future__ import annotations

from pathlib import Path

import fitz


def create_simple_rect_pdf(
    path: str | Path,
    x: float = 100.0,
    y: float = 100.0,
    w: float = 200.0,
    h: float = 150.0,
    stroke_width: float = 1.0,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Create a PDF with a single stroked rectangle.

    Args:
        path: Output file path.
        x, y: Top-left corner of the rectangle.
        w, h: Width and height.
        stroke_width: Line width.
        color: RGB stroke color (0-1 range).

    Returns:
        The output path.
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rect = fitz.Rect(x, y, x + w, y + h)
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(width=stroke_width, color=color)
    shape.commit()
    doc.save(str(path))
    doc.close()
    return path


def create_room_pdf(
    path: str | Path,
    walls: list[tuple[float, float, float, float]] | None = None,
    stroke_width: float = 1.5,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Create a PDF with multiple wall-like line segments.

    Args:
        path: Output file path.
        walls: List of (x0, y0, x1, y1) line segments. Defaults to
            a simple rectangular room.
        stroke_width: Line width for all walls.
        color: RGB stroke color.

    Returns:
        The output path.
    """
    path = Path(path)
    if walls is None:
        # Default: a rectangular room
        walls = [
            (100, 100, 400, 100),  # top wall
            (400, 100, 400, 350),  # right wall
            (400, 350, 100, 350),  # bottom wall
            (100, 350, 100, 100),  # left wall
        ]

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    for x0, y0, x1, y1 in walls:
        shape.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1))
        shape.finish(width=stroke_width, color=color)
    shape.commit()
    doc.save(str(path))
    doc.close()
    return path


def create_complex_pdf(
    path: str | Path,
) -> Path:
    """Create a PDF with walls + decorative elements for filter testing.

    Contains:
    - 4 thick black solid wall lines (a room)
    - 2 thin dashed lines (dimension annotations)
    - 1 red colored line (annotation)
    - Several short parallel lines (hatching pattern)

    Args:
        path: Output file path.

    Returns:
        The output path.
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()

    # -- Structural walls (thick, black, solid) --
    wall_segments = [
        (100, 100, 400, 100),
        (400, 100, 400, 350),
        (400, 350, 100, 350),
        (100, 350, 100, 100),
    ]
    for x0, y0, x1, y1 in wall_segments:
        shape.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1))
        shape.finish(width=2.0, color=(0, 0, 0))

    # -- Dashed dimension lines (thin, black, dashed) --
    shape.draw_line(fitz.Point(100, 80), fitz.Point(400, 80))
    shape.finish(width=0.3, color=(0, 0, 0), dashes="[3 3]")
    shape.draw_line(fitz.Point(80, 100), fitz.Point(80, 350))
    shape.finish(width=0.3, color=(0, 0, 0), dashes="[3 3]")

    # -- Red annotation line --
    shape.draw_line(fitz.Point(150, 200), fitz.Point(350, 200))
    shape.finish(width=0.5, color=(1, 0, 0))

    # -- Hatching (many short parallel lines inside the room) --
    for i in range(10):
        hx = 150 + i * 8
        shape.draw_line(fitz.Point(hx, 150), fitz.Point(hx, 300))
        shape.finish(width=0.2, color=(0, 0, 0))

    shape.commit()
    doc.save(str(path))
    doc.close()
    return path


def create_bezier_pdf(
    path: str | Path,
) -> Path:
    """Create a PDF with a cubic Bézier curve for curve-sampling tests.

    Args:
        path: Output file path.

    Returns:
        The output path.
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()

    # Draw a cubic Bézier curve (quarter circle approximation)
    p0 = fitz.Point(100, 300)
    p1 = fitz.Point(100, 190)  # control point
    p2 = fitz.Point(190, 100)  # control point
    p3 = fitz.Point(300, 100)
    shape.draw_bezier(p0, p1, p2, p3)
    shape.finish(width=1.5, color=(0, 0, 0))
    shape.commit()

    doc.save(str(path))
    doc.close()
    return path
