"""
Utility functions for the simulation.
"""
import numpy as np


def get_toroidal_distance(x1, y1, x2, y2, w, h):
    """
    Calculate toroidal (wrap-around) distance between two points.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        w: World width
        h: World height

    Returns:
        tuple: (dx, dy) - Signed distance with wrapping
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx > w / 2:
        dx = dx - w
    elif dx < -w / 2:
        dx = dx + w

    if dy > h / 2:
        dy = dy - h
    elif dy < -h / 2:
        dy = dy + h

    return dx, dy


def normalize_vector(dx, dy, max_distance):
    """
    Normalize a vector by max distance.

    Args:
        dx, dy: Vector components
        max_distance: Maximum distance for normalization

    Returns:
        tuple: Normalized (dx/max_distance, dy/max_distance)
    """
    return (dx / max_distance, dy / max_distance)


def calculate_distance_squared(x1, y1, x2, y2, toroidal=False, w=None, h=None):
    """
    Calculate squared distance between two points.

    Args:
        x1, y1: First point
        x2, y2: Second point
        toroidal: Use toroidal distance
        w, h: World dimensions (required if toroidal=True)

    Returns:
        float: Squared distance
    """
    if toroidal:
        dx, dy = get_toroidal_distance(x1, y1, x2, y2, w, h)
    else:
        dx = x2 - x1
        dy = y2 - y1

    return dx * dx + dy * dy
