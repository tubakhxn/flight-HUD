import cv2
import numpy as np
import threading
import time
import mediapipe as mp
from utils import Timer, NEON_GREEN, clamp
from gestures import GestureController
from flight_model import FlightModel
from hud import HUD

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hud = HUD(w, h)
    fm = FlightModel()

    # Shared state and detector thread (runs MediaPipe off the render loop)
    det_w, det_h = 480, 270
    shared = {
        'small_frame': None,
        'gesture_info': {'has_hand': False},
        'lock': threading.Lock(),
        'stop': False
    }

    class DetectorThread(threading.Thread):
        def __init__(self, shared, det_w, det_h):
            super().__init__(daemon=True)
            self.shared = shared
            self.det_w = det_w
            self.det_h = det_h
            self.gest = GestureController()

        def run(self):
            while not self.shared['stop']:
                frame = None
                with self.shared['lock']:
                    if self.shared['small_frame'] is not None:
                        frame = self.shared['small_frame'].copy()
                if frame is None:
                    time.sleep(0.005)
                    continue
                # process in thread-local GestureController
                results = self.gest.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                g = self.gest.analyze(results, self.det_w, self.det_h)
                with self.shared['lock']:
                    self.shared['gesture_info'] = g
                time.sleep(0.001)

    detector = DetectorThread(shared, det_w, det_h)
    detector.start()

    # mode: sky background with webcam inset (matches attached aesthetic)
    use_sky = True

    timer = Timer()

    print("Gesture Flight HUD — by TUBA KHAN. Press 'q' to quit.")
    debug_mode = False
    while True:
        dt = timer.tick()
        ret, frame = cap.read()
        if not ret:
            break

        # send a small copy to the detector thread and read last analysis
        small = cv2.resize(frame, (det_w, det_h))
        with shared['lock']:
            shared['small_frame'] = small
            g = dict(shared.get('gesture_info', {'has_hand': False}))

        # Map gestures to flight model targets smoothly
        if g.get('has_hand', False):
            fm.target_roll = g.get('roll', 0.0)
            fm.target_pitch = g.get('pitch', 0.0)
            # apply heading with a moderated multiplier for a realistic HUD feel
            fm.target_heading = (fm.target_heading + g.get('heading', 0.0) * 0.45) % 360
            fm.target_speed = clamp(200.0 + g.get('speed', 0.0), 0.0, 900.0)
            fm.target_altitude = max(0.0, fm.target_altitude + g.get('altitude', 0.0) * 0.08)
            fm.boost = g.get('boost', False)
        else:
            fm.target_roll = 0.0
            fm.target_pitch = 0.0
            fm.boost = False

        # pass whether a hand is present so the FlightModel can use more responsive smoothing
        fm.update(dt, active=g.get('has_hand', False))

        # Compose output: either sky background with webcam inset or full webcam with HUD overlay
        out_frame = np.zeros_like(frame)
        if use_sky:
            # draw HUD onto a sky background, then draw webcam inset
            hud.draw(out_frame, fm, sky_mode=True)
            web_small = cv2.resize(frame, (int(frame.shape[1] * 0.56), int(frame.shape[0] * 0.56)))
            hud.draw_webcam_inset(out_frame, web_small, label="TUBA KHAN")
        else:
            # use live webcam as background and draw HUD overlay
            out_frame[:] = frame
            hud.draw(out_frame, fm, sky_mode=False)

        # draw debug overlay optionally
        if debug_mode:
            hud.draw_debug_overlay(out_frame, fm, gesture_info=g)

        cv2.imshow('Gesture Flight HUD', out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('b'):
            # toggle background mode between sky inset and full webcam
            use_sky = not use_sky
            print(f"Sky mode: {use_sky}")
        if key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug overlay: {debug_mode}")
        if key == ord('p'):
            # cycle sky sensitivity presets (subtle -> medium -> strong)
            hud.cycle_sensitivity()
            print(f"Sky preset: {hud.sky_preset}")

    cap.release()
    # stop detector thread
    shared['stop'] = True
    detector.join(timeout=1.0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
