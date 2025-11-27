import math
from utils import smooth_damp, clamp

class FlightModel:
    def __init__(self):
        # state
        self.roll = 0.0    # degrees (-180..180)
        self.pitch = 0.0   # degrees
        self.heading = 0.0 # degrees 0..360
        self.altitude = 1000.0  # meters
        self.speed = 250.0      # knots

        # target values set by gestures
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_heading = 0.0
        self.target_altitude = self.altitude
        self.target_speed = self.speed

        # smoothing factors
        # stronger smoothing for cinematic, but still responsive
        self.smoothness = 6.5
        self.alt_smooth = 2.2
        self.speed_smooth = 3.0

        self.boost = False

    def update(self, dt, active=False):
        # Dynamic smoothing: when user has hand (active) be more responsive,
        # otherwise slow down to stabilize the aircraft.
        smooth_active = self.smoothness * (2.0 if active else 0.6)

        # Smooth roll/pitch/heading towards targets
        self.roll = smooth_damp(self.roll, self.target_roll, smooth_active, dt)
        self.pitch = smooth_damp(self.pitch, self.target_pitch, smooth_active, dt)

        # Shortest angular interpolation for heading
        diff = (self.target_heading - self.heading + 180) % 360 - 180
        self.heading = (self.heading + clamp(diff * (1 - math.exp(-smooth_active * dt)), -360, 360)) % 360

        # speed and altitude
        speed_target = self.target_speed * (1.5 if self.boost else 1.0)
        self.speed = smooth_damp(self.speed, speed_target, self.speed_smooth, dt)
        self.altitude = smooth_damp(self.altitude, self.target_altitude, self.alt_smooth, dt)

    def apply_control_deltas(self, roll_delta=0, pitch_delta=0, heading_delta=0, alt_delta=0, speed_delta=0):
        self.target_roll += roll_delta
        self.target_pitch += pitch_delta
        self.target_heading = (self.target_heading + heading_delta) % 360
        self.target_altitude += alt_delta
        self.target_speed += speed_delta
