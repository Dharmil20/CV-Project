"""
Configuration module for Lane Detection System
"""
from typing import List, Tuple

class Config:
    """
    Configuration class for lane detection parameters.
    """
    # Perspective Transform Points
    SRC_POINTS: List[Tuple[int, int]] = [(200, 720), (1100, 720), (580, 460), (700, 460)]
    DST_POINTS: List[Tuple[int, int]] = [(300, 720), (980, 720), (300, 0), (980, 0)]

    # Color & Gradient Thresholds
    S_THRESH: Tuple[int, int] = (170, 255)  # Saturation channel for yellow lines
    L_THRESH: Tuple[int, int] = (30, 255)   # Lightness channel for white lines
    SOBELX_THRESH: Tuple[int, int] = (20, 100)
    SOBEL_KERNEL: int = 15

    # Sliding Window Parameters
    NWINDOWS: int = 9
    MARGIN: int = 100
    MINPIX: int = 50
    USE_LOOK_AHEAD: bool = True
    SEARCH_MARGIN: int = 50

    # Polynomial Fitting
    POLY_ORDER: int = 2  # 2nd order for curves

    # Metrics Conversion
    YM_PER_PIX: float = 30/720
    XM_PER_PIX: float = 3.7/700

    # Visualization
    SHOW_BINARY: bool = False
    SHOW_WARPED: bool = False
    SHOW_WINDOWS: bool = False
    SHOW_METRICS: bool = True
    LANE_ALPHA: float = 0.3
    SMOOTHING_FRAMES: int = 5
