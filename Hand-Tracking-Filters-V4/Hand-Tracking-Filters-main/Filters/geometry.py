import cv2
import numpy as np


def portal_width(p1, p2, p3, p4):
    top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
    bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
    return (top_w + bottom_w) / 2.0


class PointSmoother:
    """Exponential smoother that reduces hand-jitter without adding much lag."""

    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.points = None

    def update(self, points):
        current = np.asarray(points, dtype=np.float32)
        if self.points is None or self.points.shape != current.shape:
            self.points = current.copy()
        else:
            self.points = self.alpha * current + (1.0 - self.alpha) * self.points
        return [tuple(p) for p in self.points]

    def reset(self):
        self.points = None


class ClosingGestureDetector:
    """Hysteresis + cooldown prevents repeated filter changes from one gesture."""

    def __init__(self, close_ratio=0.16, open_ratio=0.30, cooldown_frames=18):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.cooldown_frames = cooldown_frames
        self.is_closed = False
        self.cooldown = 0

    def update(self, width, frame_w):
        if self.cooldown > 0:
            self.cooldown -= 1

        close_threshold = self.close_ratio * frame_w
        open_threshold = self.open_ratio * frame_w

        triggered = False
        if not self.is_closed and width < close_threshold and self.cooldown == 0:
            self.is_closed = True
            self.cooldown = self.cooldown_frames
            triggered = True
        elif self.is_closed and width > open_threshold:
            self.is_closed = False

        return triggered


def paint_filter_in_polygon(frame, polygon_pts, filter_func):
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(polygon_pts)
    x = max(x, 0)
    y = max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return frame

    roi = frame[y:y + bh, x:x + bw]
    filtered_roi = filter_func(roi)

    # Build the mask only for the bounding rectangle rather than the whole frame.
    local_polygon = polygon_pts.astype(np.int32) - np.array([x, y], dtype=np.int32)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)

    cv2.copyTo(filtered_roi, mask, roi)
    return frame


def render_portal(frame, p1, p2, p3, p4, filter_func):
    full_polygon = np.array([p1, p3, p4, p2], dtype=np.float32)
    polygon_int = np.rint(full_polygon).astype(np.int32)

    paint_filter_in_polygon(frame, polygon_int, filter_func)
    cv2.polylines(frame, [polygon_int], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Small corner accents make the portal easier to see without a heavy overlay.
    for px, py in polygon_int:
        cv2.circle(frame, (int(px), int(py)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return frame
