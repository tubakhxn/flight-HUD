import math
import numpy as np
import mediapipe as mp
from utils import vec2, length, angle_between, clamp, lerp

mp_hands = mp.solutions.hands


class GestureController:
    def __init__(self, max_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=max_hands,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)
        self.last = None
        self.out_smooth = 0.55

    def process(self, image_rgb):
        return self.hands.process(image_rgb)

    def analyze(self, results, image_w, image_h):
        out = {
            'roll': 0.0, 'pitch': 0.0, 'heading': 0.0, 'speed': 0.0,
            'altitude': 0.0, 'boost': False, 'has_hand': False
        }

        if not results or not getattr(results, 'multi_hand_landmarks', None):
            return out

        lm = results.multi_hand_landmarks[0]
        pts = []
        for p in lm.landmark:
            pts.append((p.x * image_w, p.y * image_h, p.z))

        # Indices
        WRIST = 0
        INDEX_TIP = 8
        THUMB_TIP = 4
        MIDDLE_MCP = 9
        PINKY_MCP = 17

        cx = sum([p[0] for p in pts]) / len(pts)
        cy = sum([p[1] for p in pts]) / len(pts)
        cz = sum([p[2] for p in pts]) / len(pts)

        out['has_hand'] = True

        # Roll: angle of mid-knuckle relative to wrist
        v_mid = vec2(pts[MIDDLE_MCP][0] - pts[WRIST][0], pts[MIDDLE_MCP][1] - pts[WRIST][1])
        roll_angle = math.degrees(math.atan2(v_mid[1], v_mid[0])) - 90
        out['roll'] = -roll_angle

        # Pitch: vertical position + wrist tilt
        rel_y = (cy - image_h / 2) / (image_h / 2)
        wrist_vec = vec2(pts[INDEX_TIP][0] - pts[WRIST][0], pts[INDEX_TIP][1] - pts[WRIST][1])
        wrist_tilt = math.degrees(math.atan2(-wrist_vec[1], wrist_vec[0]))
        out['pitch'] = clamp(-rel_y * 30.0 + (wrist_tilt - 90) * 0.2, -45, 45)

        # Speed (depth proxy): wrist z and hand span
        span = length(vec2(pts[THUMB_TIP][0] - pts[PINKY_MCP][0], pts[THUMB_TIP][1] - pts[PINKY_MCP][1]))
        out['speed'] = (-cz) * 600.0 + (span / image_w) * 200.0

        # Heading: combine twist (rotation of fingers) with lateral hand position
        v1 = vec2(pts[INDEX_TIP][0] - pts[WRIST][0], pts[INDEX_TIP][1] - pts[WRIST][1])
        v2 = vec2(pts[MIDDLE_MCP][0] - pts[WRIST][0], pts[MIDDLE_MCP][1] - pts[WRIST][1])
        heading_delta = angle_between(v1, v2) * 0.6
        # lateral normalized (-1..1)
        # lateral normalized (-1..1). Use inverted sign to match on-screen movement
        rel_x = (cx - image_w / 2) / (image_w / 2)
        # apply a much smaller lateral contribution for a realistic HUD feel
        lateral_contrib = -rel_x * 15.0
        out['heading'] = heading_delta + lateral_contrib
        out['rel_x'] = rel_x

        out['altitude'] = -rel_y * 40.0

        # Pinch for boost
        d = length(vec2(pts[THUMB_TIP][0] - pts[INDEX_TIP][0], pts[THUMB_TIP][1] - pts[INDEX_TIP][1]))
        pinch_thresh = image_w * 0.04
        out['boost'] = d < pinch_thresh

        # clamp ranges
        out['roll'] = max(-70, min(70, out['roll']))
        out['pitch'] = max(-45, min(45, out['pitch']))
        out['heading'] = max(-90, min(90, out['heading']))
        out['speed'] = max(-300, min(900, out['speed']))
        out['altitude'] = max(-200, min(200, out['altitude']))

        # deadzones
        def apply_deadzone(v, dz):
            return 0.0 if abs(v) < dz else v

        out['roll'] = apply_deadzone(out['roll'], 1.5)
        out['pitch'] = apply_deadzone(out['pitch'], 0.8)
        out['heading'] = apply_deadzone(out['heading'], 0.8)
        out['speed'] = apply_deadzone(out['speed'], 8.0)

        # smoothing
        if self.last is None:
            self.last = out.copy()
        else:
            for k in ('roll', 'pitch', 'heading', 'speed', 'altitude'):
                self.last[k] = lerp(self.last[k], out[k], self.out_smooth)
            self.last['boost'] = out['boost']
            self.last['has_hand'] = out['has_hand']
            out = dict(self.last)

        return out
