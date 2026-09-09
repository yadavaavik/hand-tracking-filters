import time
from pathlib import Path

import cv2
import mediapipe as mp

from hand_tracking import pinch_ratio
from geometry import render_portal, portal_width, ClosingGestureDetector, PointSmoother
from filters import FILTERS, FILTER_NAMES

WINDOW_NAME = "Hand Tracking Filters"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class AutoBrightness:
    """Adaptive brightness/contrast for more stable hand detection."""

    def __init__(self, target_mean=125.0, min_gain=0.78, max_gain=1.65, smoothing=0.10):
        self.target_mean = target_mean
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.smoothing = smoothing
        self.gain = 1.0
        self.clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))

    def apply(self, frame):
        small = cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = float(gray_small.mean())

        desired = self.target_mean / max(mean, 25.0)
        desired = max(self.min_gain, min(self.max_gain, desired))
        self.gain = (1.0 - self.smoothing) * self.gain + self.smoothing * desired

        adjusted = cv2.convertScaleAbs(frame, alpha=self.gain, beta=0)

        # Mild local contrast enhancement, created once in __init__.
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def put_small_panel(frame, text, x, y, align="left"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if align == "right":
        x -= tw
    x1, y1 = x - 7, y - th - 6
    x2, y2 = x + tw + 7, y + baseline + 6

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def save_screenshot(frame, counter):
    out_dir = Path.cwd() / "screenshots"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"filter_{counter:03d}.png"
    cv2.imwrite(str(path), frame)
    return path


def any_pinch_tapped(hands_list, pinch_state, cooldown):
    """Trigger once when either hand enters a thumb-index pinch."""
    cooldown = max(0, cooldown - 1)
    triggered = False
    active_ids = set()

    for hand_id, hand_landmarks in enumerate(hands_list):
        active_ids.add(hand_id)
        ratio = pinch_ratio(hand_landmarks)
        is_pinched = pinch_state.get(hand_id, False)

        # Hysteresis prevents jitter around the threshold.
        if not is_pinched and ratio < 0.58:
            pinch_state[hand_id] = True
            if cooldown == 0:
                triggered = True
                cooldown = 14
        elif is_pinched and ratio > 0.82:
            pinch_state[hand_id] = False

    for hand_id in list(pinch_state):
        if hand_id not in active_ids:
            pinch_state.pop(hand_id, None)

    return triggered, cooldown


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.60,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Check camera permissions or index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    filter_index = 0
    closing_detector = ClosingGestureDetector(cooldown_frames=20)
    smoother = PointSmoother(alpha=0.24)
    brightness = AutoBrightness()
    pinch_state = {}
    pinch_cooldown = 0
    screenshot_counter = 1
    fps_smooth = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps_smooth = current_fps if fps_smooth == 0 else (0.90 * fps_smooth + 0.10 * current_fps)

            frame = brightness.apply(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            left_hand = None
            right_hand = None
            detected_hands = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    raw_label = handedness.classification[0].label
                    label = "Right" if raw_label == "Left" else "Left"
                    detected_hands.append(hand_landmarks)
                    if label == "Left":
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks

            tapped, pinch_cooldown = any_pinch_tapped(detected_hands, pinch_state, pinch_cooldown)
            if tapped:
                filter_index = (filter_index + 1) % len(FILTERS)

            if left_hand is not None and right_hand is not None:
                lm_left = left_hand.landmark
                lm_right = right_hand.landmark
                raw_points = [
                    (lm_left[8].x * w, lm_left[8].y * h),
                    (lm_left[4].x * w, lm_left[4].y * h),
                    (lm_right[8].x * w, lm_right[8].y * h),
                    (lm_right[4].x * w, lm_right[4].y * h),
                ]
                p1, p2, p3, p4 = smoother.update(raw_points)

                width = portal_width(p1, p2, p3, p4)
                if closing_detector.update(width, w):
                    filter_index = (filter_index + 1) % len(FILTERS)

                frame = render_portal(frame, p1, p2, p3, p4, FILTERS[filter_index])
            else:
                smoother.reset()

            # Minimal HUD: only current filter + FPS.
            put_small_panel(frame, FILTER_NAMES[filter_index], 10, h - 12)
            put_small_panel(frame, f"{fps_smooth:.0f} FPS", w - 10, 24, align="right")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                saved = save_screenshot(frame, screenshot_counter)
                screenshot_counter += 1
                print(f"Screenshot saved: {saved}")
            elif key in (ord(" "), ord("d"), ord("D")):
                filter_index = (filter_index + 1) % len(FILTERS)
            elif key in (ord("a"), ord("A")):
                filter_index = (filter_index - 1) % len(FILTERS)
            elif key == 81:  # left arrow on some OpenCV builds
                filter_index = (filter_index - 1) % len(FILTERS)
            elif key == 83:  # right arrow on some OpenCV builds
                filter_index = (filter_index + 1) % len(FILTERS)

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
