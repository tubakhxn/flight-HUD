import cv2
import math
import numpy as np
from utils import NEON_GREEN, NEON_GREEN_FADED, lerp

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def generate_sky(width, height, t=0.0):
    """Generate a moving twilight sky (blue/purple) similar to the reference.

    This uses a vertical gradient with a pinkish horizon band, layered blurred
    cloud textures, and a controlled number of stars. Output is BGR (uint8).
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)

    # Color stops in BGR (top, mid, horizon) - tuned for lighter blue sky
    # Colors expressed in BGR
    top = np.array([140, 60, 20], dtype=float)    # deep blue (B,G,R)
    mid = np.array([200, 110, 50], dtype=float)   # lighter blue
    horizon = np.array([220, 150, 90], dtype=float) # pale horizon (cooler, not pink)

    # blend vertical gradient with slight horizontal variation
    blend = np.clip(1.0 - (yv ** 1.1), 0.0, 1.0)
    sky = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        sky[:, :, c] = top[c] * blend + mid[c] * (1.0 - blend) * 0.6 + horizon[c] * (1.0 - blend) * 0.4

    # add a soft horizon band (reduced warmth to avoid pink tint)
    horizon_band = np.exp(-((yv - 0.75) ** 2) / 0.008)  # peaked near lower 25%
    for c in range(3):
        sky[:, :, c] = np.clip(sky[:, :, c] + horizon_band * (horizon[c] * 0.18), 0, 255)

    # layered cloud textures (coarse + fine) using sin + gaussian blur
    layer_coarse = (np.sin((xv * 2.1 + yv * 1.2) * 3.1415 + t * 0.15) + 1) * 0.5
    layer_fine = (np.sin((xv * 9.0 + yv * 4.0) * 3.1415 + t * 0.35) + 1) * 0.5
    coarse = cv2.GaussianBlur((layer_coarse * 255).astype(np.uint8), (0, 0), sigmaX=40)
    fine = cv2.GaussianBlur((layer_fine * 255).astype(np.uint8), (0, 0), sigmaX=10)
    cloud = np.clip(0.05 * coarse.astype(np.float32) + 0.02 * fine.astype(np.float32), 0, 255)
    for c in range(3):
        # blend clouds subtly, keep cooler tones
        sky[:, :, c] = np.clip(sky[:, :, c] * (1 - 0.10) + cloud * 0.10, 0, 255)

    # stars (fewer, softer)
    rng = np.random.RandomState(int(t * 100) & 0xFFFF)
    num_stars = 70
    for i in range(num_stars):
        sx = int(rng.rand() * width)
        sy = int(rng.rand() * height * 0.55)
        brightness = rng.uniform(190, 255)
        rr = rng.randint(1, 2)
        cv2.circle(sky, (sx, sy), rr, (brightness, brightness, brightness), -1)

    return sky.astype(np.uint8)

class Particle:
    def __init__(self, w, h):
        self.w = w; self.h = h
        self.reset()

    def reset(self):
        self.x = np.random.uniform(0, self.w)
        self.y = np.random.uniform(0, self.h)
        self.vx = np.random.uniform(-5, 5) * 0.02
        self.vy = np.random.uniform(-5, 5) * 0.02
        self.size = np.random.uniform(1, 3)
        self.alpha = np.random.uniform(0.08, 0.3)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > self.w or self.y < 0 or self.y > self.h:
            self.reset()

class HUD:
    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.center = (self.w // 2, self.h // 2)
        # fewer particles for performance
        self.particles = [Particle(self.w, self.h) for _ in range(36)]
        self.time = 0.0
        self.debug = False
        # optional font path for sci-fi monospaced font placed at resources/Orbitron-Regular.ttf
        self.font_path = None
        # Try to auto-find a bundled font file
        import os
        bundled = os.path.join(os.path.dirname(__file__), 'resources', 'Orbitron-Regular.ttf')
        if os.path.exists(bundled):
            self.font_path = bundled
        # Sky sensitivity presets (roll scale in degrees->rotation multiplier, pitch scale in px/deg)
        # Choose a preset that closely matches the reference by default.
        self.sky_presets = {
            'subtle': {'roll': 0.25, 'pitch': 3.0},
            'medium': {'roll': 0.45, 'pitch': 5.0},
            'strong': {'roll': 0.75, 'pitch': 8.0},
        }
        # Default to a stronger preset to match the reference's visible parallax
        self.sky_preset = 'strong'
        self.sky_roll_scale = self.sky_presets[self.sky_preset]['roll']
        self.sky_pitch_scale = self.sky_presets[self.sky_preset]['pitch']

    def set_sensitivity(self, preset_name: str):
        if preset_name in self.sky_presets:
            self.sky_preset = preset_name
            p = self.sky_presets[preset_name]
            self.sky_roll_scale = p['roll']
            self.sky_pitch_scale = p['pitch']

    def cycle_sensitivity(self):
        keys = list(self.sky_presets.keys())
        idx = keys.index(self.sky_preset)
        idx = (idx + 1) % len(keys)
        self.set_sensitivity(keys[idx])

    def _prepare_sky(self):
        # Precompute a single sky texture and a blurred version used for glow
        if hasattr(self, 'sky_base') and self.sky_base is not None:
            return
        self.sky_base = generate_sky(self.w, self.h, t=0.0)
        # blurred for glow/soft overlay
        self.sky_blur = cv2.GaussianBlur(self.sky_base, (0, 0), sigmaX=18)

    def draw(self, frame, flight_state, sky_mode=True):
        self._prepare_sky()
        # If sky_mode is True we draw generated sky background and place webcam inset;
        # otherwise, overlay is the live webcam full frame.
        overlay = frame.copy()
        self.time += 0.033
        if sky_mode:
            # animated pan/shift of precomputed sky
            # base drift keeps motion alive; heading provides left/right control
            base_drift = int((self.time * 6) % self.w)
            # normalize heading to -180..180 then map to pixels so 0 heading -> centered sky
            hnorm = ((float(flight_state.heading) + 180.0) % 360.0) - 180.0
            # soften sky pan so heading changes move the sky only gently
            heading_pan = int(hnorm * 0.45)
            shift_x = (base_drift + heading_pan) % self.w
            base = np.roll(self.sky_base, shift_x, axis=1)
            blur = np.roll(self.sky_blur, shift_x, axis=1)

            # responsive sky warp: small rotation driven by roll and vertical shift driven by pitch
            # roll -> gentle rotation (half-scale), pitch -> vertical offset (pixels)
            try:
                cx, cy = self.center
                # scale rotation and vertical shift based on chosen preset
                angle = -float(flight_state.roll) * self.sky_roll_scale  # degrees
                dy = int(-float(flight_state.pitch) * self.sky_pitch_scale)
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                M[1, 2] += dy
                base_warp = cv2.warpAffine(base, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                blur_warp = cv2.warpAffine(blur, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            except Exception:
                base_warp = base
                blur_warp = blur

            # add slight shimmer by blending with blurred texture
            shimmer = (0.06 + 0.02 * math.sin(self.time * 0.6))
            overlay = cv2.addWeighted(base_warp, 1.0 - shimmer, blur_warp, shimmer, 0)

        self._draw_particles(overlay)
        self._draw_artificial_horizon(overlay, flight_state)
        self._draw_bank_arc(overlay, flight_state)
        self._draw_pitch_ladder(overlay, flight_state)
        self._draw_heading(overlay, flight_state)
        self._draw_tapes(overlay, flight_state)
        self._draw_center_reticle(overlay)
        # lightweight bloom/glow: blur bright neon areas and add back for glow
        try:
            neon_mask = (overlay[:, :, 1] > 90) & (overlay[:, :, 0] < 140)
            neon_only = np.zeros_like(overlay)
            neon_only[neon_mask] = overlay[neon_mask]
            glow = cv2.GaussianBlur(neon_only, (0, 0), sigmaX=9)
            overlay = cv2.addWeighted(overlay, 0.9, glow, 0.35, 0)
        except Exception:
            pass
        # Note: webcam inset drawing is handled externally when using sky_mode.
        # Blend overlay onto provided frame for subtle HUD glow when in full-webcam mode.
        if sky_mode:
            # return overlay as final composited background
            frame[:, :] = overlay
        else:
            cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

    def _draw_particles(self, img):
        # fewer particles; draw as small points for performance
        for p in self.particles:
            p.step()
            x, y = int(p.x), int(p.y)
            c = (int(NEON_GREEN[0]*p.alpha), int(NEON_GREEN[1]*p.alpha), int(NEON_GREEN[2]*p.alpha))
            if 0 <= y < self.h and 0 <= x < self.w:
                img[y, x] = c

    def _draw_artificial_horizon(self, img, s):
        cx, cy = self.center
        roll = math.radians(s.roll)
        pitch = s.pitch

        # Draw rotated horizon line
        length = max(self.w, self.h) * 1.5
        # vertical offset from pitch (scale pitch to pixels)
        pitch_px = pitch * 8
        x1 = int(cx - math.cos(roll) * length + math.sin(roll) * pitch_px)
        y1 = int(cy - math.sin(roll) * length - math.cos(roll) * pitch_px)
        x2 = int(cx + math.cos(roll) * length + math.sin(roll) * pitch_px)
        y2 = int(cy + math.sin(roll) * length - math.cos(roll) * pitch_px)

        cv2.line(img, (x1, y1), (x2, y2), NEON_GREEN, 1)

    def _draw_bank_arc(self, img, s):
        cx, cy = self.center
        # arc above center for bank indicator
        radius = 160
        # Draw markers
        for ang in range(-60, 61, 10):
            a = math.radians(ang - s.roll)
            x = int(cx + math.sin(a) * radius)
            y = int(cy - math.cos(a) * radius - 220)
            # major markers
            if ang % 30 == 0:
                cv2.line(img, (x - 6, y), (x + 6, y), NEON_GREEN, 1)
            else:
                cv2.line(img, (x - 3, y), (x + 3, y), NEON_GREEN, 1)
        # center triangle (slightly smaller for thinner look)
        cv2.drawMarker(img, (cx, cy - 220 - radius - 10), NEON_GREEN, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=12, thickness=1)

    def _draw_pitch_ladder(self, img, s):
        cx, cy = self.center
        roll = math.radians(s.roll)
        pitch = s.pitch
        # draw multiple lines above/below
        for step in range(-60, 61, 5):
            y_off = (step - pitch) * 8
            length = 260 if step % 10 == 0 else 120
            a = roll
            x1 = int(cx - math.cos(a) * length + math.sin(a) * y_off)
            y1 = int(cy - math.sin(a) * length - math.cos(a) * y_off)
            x2 = int(cx + math.cos(a) * length + math.sin(a) * y_off)
            y2 = int(cy + math.sin(a) * length - math.cos(a) * y_off)
            color = NEON_GREEN if abs(step) % 10 == 0 else NEON_GREEN_FADED
            thickness = 1
            if 0 <= y1 < self.h or 0 <= y2 < self.h:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                if abs(step) % 10 == 0:
                    # smaller numeric labels
                    cv2.putText(img, f"{step:+}", (x2 + 8, y2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_GREEN, 1, cv2.LINE_AA)

    def _draw_heading(self, img, s):
        cx, cy = self.center
        # heading tape at bottom
        tape_y = self.h - 80
        # draw center heading (use ASCII 'deg' to avoid platform-specific font issues)
        cv2.putText(img, f"HDG {int(s.heading):03}deg", (cx - 60, tape_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.95, NEON_GREEN, 1, cv2.LINE_AA)
        # draw surrounding tick marks
        for offset in range(-50, 51, 10):
            hdg = (s.heading + offset) % 360
            x = int(cx + offset * 4.0)
            cv2.line(img, (x, tape_y - 8), (x, tape_y + 8), NEON_GREEN, 1)
            if offset % 30 == 0:
                cv2.putText(img, f"{int(hdg):03}", (x - 16, tape_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_GREEN, 1, cv2.LINE_AA)

    def _draw_tapes(self, img, s):
        # Speed tape left
        left_x = 60
        cy = self.center[1]
        # speed scale
        for i in range(-5, 6):
            val = int(s.speed) + i * 10
            y = int(cy + i * 18)
            color = NEON_GREEN if i == 0 else NEON_GREEN_FADED
            cv2.putText(img, str(val), (left_x - 28, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # altitude tape right
        right_x = self.w - 120
        for i in range(-5, 6):
            val = int(s.altitude) + i * 50
            y = int(cy + i * 18)
            color = NEON_GREEN if i == 0 else NEON_GREEN_FADED
            cv2.putText(img, str(val), (right_x, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_center_reticle(self, img):
        cx, cy = self.center
        # concentric rings and crosshair
        cv2.circle(img, (cx, cy), 60, NEON_GREEN, 1)
        cv2.line(img, (cx - 90, cy), (cx - 30, cy), NEON_GREEN, 1)
        cv2.line(img, (cx + 90, cy), (cx + 30, cy), NEON_GREEN, 1)
        cv2.line(img, (cx, cy - 90), (cx, cy - 30), NEON_GREEN, 1)
        cv2.line(img, (cx, cy + 90), (cx, cy + 30), NEON_GREEN, 1)
        cv2.putText(img, "TARGET", (cx - 42, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_GREEN, 1, cv2.LINE_AA)

    def _draw_webcam_box(self, img):
        # kept for legacy; prefer drawing inset via `draw_webcam_inset`
        pass

    def draw_webcam_inset(self, canvas, webcam_frame, label="TUBA KHAN"):
        # Draw stylized neon green inset window on `canvas` using `webcam_frame` resized
        inset_h = int(self.h * 0.36)
        inset_w = int(inset_h * webcam_frame.shape[1] / webcam_frame.shape[0])
        pad = 14
        x0 = self.w - inset_w - 60
        y0 = 60
        x1 = x0 + inset_w
        y1 = y0 + inset_h

        # framed polygon (cut corner)
        pts = np.array([[x0, y0], [x1 - 28, y0], [x1, y0 + 28], [x1, y1], [x0, y1]], np.int32)
        cv2.fillPoly(canvas, [pts], (0, 0, 0))

        # compute region dimensions (leave a 2px inner border)
        region_x0 = x0 + 2
        region_y0 = y0 + 2
        region_x1 = x1 - 2
        region_y1 = y1 - 2
        region_w = max(1, region_x1 - region_x0)
        region_h = max(1, region_y1 - region_y0)

        # Resize webcam to fit exactly the inner region
        web = cv2.resize(webcam_frame, (region_w, region_h))

        # place webcam image into canvas inner region
        canvas[region_y0:region_y1, region_x0:region_x1] = web

        # draw neon outline (thinner)
        cv2.polylines(canvas, [pts], True, NEON_GREEN, 2, cv2.LINE_AA)

        # label box above (use our text helper for tighter rendering if PIL available)
        lab_w = int(inset_w * 0.7)
        lab_x0 = x0
        lab_y0 = y0 - 36
        lab_x1 = lab_x0 + lab_w
        lab_y1 = y0 - 8
        cv2.rectangle(canvas, (lab_x0, lab_y0), (lab_x1, lab_y1), NEON_GREEN, 2)
        # draw label using _draw_text which prefers PIL + ttf if available for tighter spacing
        try:
            self._draw_text(canvas, label, (lab_x0 + 8, lab_y0 + 6), size=20, color=NEON_GREEN)
        except Exception:
            cv2.putText(canvas, label, (lab_x0 + 8, lab_y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, NEON_GREEN, 1, cv2.LINE_AA)

        # small caption below inset
        try:
            self._draw_text(canvas, "control mode", (x0 + 8, y1 + 8), size=14, color=NEON_GREEN_FADED)
        except Exception:
            cv2.putText(canvas, "control mode", (x0 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_GREEN_FADED, 1, cv2.LINE_AA)

    def _draw_text(self, img, text, pos, size=18, color=NEON_GREEN):
        x, y = pos
        if PIL_AVAILABLE and self.font_path:
            try:
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(self.font_path, size)
                # convert color from BGR to RGB
                rgb = (int(color[2]), int(color[1]), int(color[0]))
                draw.text((x, y), text, font=font, fill=rgb)
                img[:, :] = np.array(img_pil)
                return
            except Exception:
                pass

        # fallback to OpenCV
        cv2.putText(img, text, (x, y + size // 2), cv2.FONT_HERSHEY_DUPLEX, size / 24.0, color, 1, cv2.LINE_AA)

    def draw_debug_overlay(self, img, flight_state, gesture_info=None):
        # Draw numeric readouts for tuning gestures and flight state
        x0, y0 = 18, 18
        lines = [
            f"ROLL: {flight_state.roll:6.2f} deg",
            f"PITCH: {flight_state.pitch:6.2f} deg",
            f"HDG: {flight_state.heading:6.1f} deg",
            f"SPEED: {flight_state.speed:6.1f} kt",
            f"ALT: {flight_state.altitude:6.1f} m",
            f"BOOST: {flight_state.boost}",
        ]
        # include sky preset info
        lines.insert(0, f"SKY PRESET: {self.sky_preset.upper()}")
        if gesture_info:
            lines.append("--- GESTURE ---")
            for k in ('roll', 'pitch', 'heading', 'speed', 'altitude', 'boost'):
                if k in gesture_info:
                    lines.append(f"{k.upper()}: {str(gesture_info[k])[:7]}")

        for i, l in enumerate(lines):
            self._draw_text(img, l, (x0, y0 + i * 22), size=18, color=NEON_GREEN)
