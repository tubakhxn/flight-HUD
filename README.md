# Gesture Flight HUD

Simple Flight HUD — a compact, webcam-driven HUD prototype. Use your hands to control a neon HUD and a lightweight flight model.

Creator: @tubakhxn

---

## What is this

Gesture Flight HUD is a small demo/prototype that turns your webcam into a hands-only flight HUD. It uses MediaPipe Hands to detect one hand, maps gestures to a smoothed flight state (roll, pitch, heading, speed, altitude, boost), and renders a neon HUD overlay on a procedurally generated sky using OpenCV.

This project is ideal for demos, experiments, and quick prototypes where you want a cinematic UI that responds to hand motion without extra hardware.

---

## Features

- Real-time hand detection (MediaPipe)
- Gesture -> flight controls (roll, pitch, heading, speed, altitude, pinch boost)
- Smooth flight model and HUD rendering (OpenCV)
- Sky presets, HUD glow, webcam inset with label
- Lightweight, easy to fork and extend

---

## Quickstart

Prerequisites: Python 3.8+ and a webcam.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the demo:

```powershell
cd 'C:\Users\Tuba Khan\Downloads\plane'
python main.py
```

Keyboard controls (while window is focused):
- `p` — cycle sky sensitivity presets (subtle / medium / strong)
- `d` — toggle debug overlay (shows numeric gesture values)
- `b` — toggle sky-with-webcam-inset vs full-webcam background
- `q` or `Esc` — quit

---

## How gestures map to controls

- Roll: wrist rotation (bank left/right)
- Pitch: vertical hand position + wrist tilt (nose up/down)
- Heading: lateral hand position + finger twist (left/right turns)
- Speed: depth proxy (wrist z) + hand span
- Altitude: large vertical palm movement
- Boost: pinch (thumb + index)

Use the `d` debug overlay to see live numeric values while testing gestures.

---

## Forking & contribution

Want to fork this project and build on it? Great!

1. Click `Fork` on the GitHub repo page (or clone then create your own repository).
2. Create a branch for your changes: `git checkout -b feature/my-change`
3. Make changes, run tests / demo locally, and commit: `git commit -am "Add feature"`
4. Push and open a Pull Request to the original repo.

Be sure to include a short description of why you made the change and any runtime expectations.

---

## License

This project is released under the MIT License — a short, permissive license that lets people use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.

Full text (MIT License):

```
MIT License

Copyright (c) 2025 @tubakhxn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Credits

- Creator / Developed by: `@tubakhxn`
- Uses Google MediaPipe Hands (for hand detection)
- Uses OpenCV (for rendering and composition)
- Uses NumPy (numerical helpers)

If you used external assets (fonts, images), make sure to credit and include their licenses in `resources/`.

---

## Troubleshooting

- If the app is slow: try lowering detection resolution in `main.py` (`det_w, det_h = 320, 180`) or reduce bloom sigma in `hud.py`.
- If MediaPipe fails to initialize: ensure you have a compatible Python and the `mediapipe` package installed.
- If the webcam inset is mirrored/wrong orientation: adjust your camera settings or flip the frame in `main.py` before sending to the detector.

---

If you want, I can also add a short demo script and a 30s spoken narration you can use for a video. Tell me if you want that next.# Gesture Flight HUD — by TUBA KHAN

Live gesture-controlled aircraft HUD using OpenCV + MediaPipe Hands.

Requirements
- Python 3.8+
- Install dependencies:
```
pip install -r requirements.txt
```

Run
```
python main.py
```

Toggle
- Press `b` to toggle between the generated sky background with a webcam inset (default,
  matching the attached aesthetic) and using the full webcam feed as the background.

Debug / Calibration
- Press `d` to toggle a debug overlay that shows numeric flight-state readouts and the
    current gesture values (helpful for tuning sensitivity and mapping).
- To use a custom sci‑fi monospaced TTF for the inset label and debug text, place the
    font file at `resources/Orbitron-Regular.ttf` (create a `resources` folder next to the
    script). The program will use the TTF if available and fall back to OpenCV fonts otherwise.

Controls (via hand gestures in front of webcam):
- Tilt hand left/right → roll (bank)
- Move hand up/down → pitch
- Move hand forward/back (z) → speed
- Rotate wrist (twist) → heading
- Pinch (thumb+index) → boost mode

Notes
- Keep one hand visible to the camera. Lighting helps MediaPipe detection.
- The HUD overlays live webcam feed with neon-green Top-Gun-style instruments.
