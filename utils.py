import math
import time
import numpy as np

# Colors and constants
NEON_GREEN = (78, 255, 84)  # BGR
NEON_GREEN_FADED = (40, 140, 50)
HUD_ALPHA = 0.95

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, a, b):
    return max(a, min(b, v))

def smooth_damp(current, target, smoothing, dt):
    # exponential smoothing
    if dt <= 0:
        return target
    t = 1.0 - math.exp(-smoothing * dt)
    return lerp(current, target, t)

def vec2(x, y):
    return np.array([x, y], dtype=float)

def length(v):
    return math.hypot(v[0], v[1])

def angle_between(v1, v2):
    # returns signed angle in degrees v1 -> v2
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    return math.degrees(a2 - a1)

class Timer:
    def __init__(self):
        self.last = time.time()

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        return dt
