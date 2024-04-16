import numpy as np

def cosd(angle):
    """Compute cosine of an angle in degrees."""
    return np.cos(np.radians(angle))

def sind(angle):
    """Compute sine of an angle in degrees."""
    return np.sin(np.radians(angle))

def tand(angle):
    """Compute tangent of an angle in degrees."""
    return np.tan(np.radians(angle))

def acosd(value):
    """Compute angle from cosine value in degrees."""
    return np.degrees(np.arccos(value))
