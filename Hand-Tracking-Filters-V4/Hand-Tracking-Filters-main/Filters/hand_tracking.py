import numpy as np

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17


def _dist(lm, i, j, w, h):
    a = np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
    b = np.array([lm[j].x * w, lm[j].y * h], dtype=np.float32)
    return float(np.linalg.norm(a - b))


def get_extended_fingers(hand_landmarks, w, h):
    lm = hand_landmarks.landmark

    def is_extended(tip, mcp):
        return _dist(lm, tip, WRIST, w, h) > _dist(lm, mcp, WRIST, w, h) * 1.3

    return {
        "thumb": is_extended(THUMB_TIP, THUMB_MCP),
        "index": is_extended(INDEX_TIP, INDEX_MCP),
        "middle": is_extended(MIDDLE_TIP, MIDDLE_MCP),
        "ring": is_extended(RING_TIP, RING_MCP),
        "pinky": is_extended(PINKY_TIP, PINKY_MCP),
    }


def pinch_ratio(hand_landmarks):
    """Thumb-index distance normalized by the hand's palm width.

    Lower values mean the thumb and index finger are closer together.
    This normalization makes pinch detection work at different distances
    from the camera and across different hand sizes.
    """
    lm = hand_landmarks.landmark
    thumb = np.array([lm[THUMB_TIP].x, lm[THUMB_TIP].y], dtype=np.float32)
    index = np.array([lm[INDEX_TIP].x, lm[INDEX_TIP].y], dtype=np.float32)
    palm = np.array([lm[5].x, lm[5].y], dtype=np.float32)
    wrist = np.array([lm[WRIST].x, lm[WRIST].y], dtype=np.float32)
    palm_width = max(float(np.linalg.norm(palm - wrist)), 1e-4)
    return float(np.linalg.norm(thumb - index) / palm_width)
