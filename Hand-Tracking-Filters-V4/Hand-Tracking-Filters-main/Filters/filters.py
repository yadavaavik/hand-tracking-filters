import cv2
import numpy as np

_MESH_CACHE = {}
_VIGNETTE_CACHE = {}
_NOISE_CACHE = {}


def _meshgrid(h, w):
    key = (h, w)
    cached = _MESH_CACHE.get(key)
    if cached is None:
        yy, xx = np.indices((h, w), dtype=np.float32)
        _MESH_CACHE[key] = (yy, xx)
        cached = (yy, xx)
    return cached


def _vignette(h, w):
    key = (h, w)
    cached = _VIGNETTE_CACHE.get(key)
    if cached is None:
        yy, xx = _meshgrid(h, w)
        cy, cx = h / 2.0, w / 2.0
        max_dist = max(np.hypot(cx, cy), 1.0)
        cached = np.clip(1.0 - 0.5 * (np.hypot(xx - cx, yy - cy) / max_dist), 0.0, 1.0)[..., None]
        _VIGNETTE_CACHE[key] = cached.astype(np.float32)
    return cached


def filter_1(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = np.empty_like(roi)
    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)
    return out


def _dot_filter(roi, cell, base_color, dot_color, radius_scale):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = _meshgrid(h, w)
    cx = (xx % cell) - cell / 2.0
    cy = (yy % cell) - cell / 2.0
    dist2 = cx * cx + cy * cy
    radius = (1.0 - gray.astype(np.float32) / 255.0) * (cell / radius_scale)
    out = np.full_like(roi, base_color)
    out[dist2 < radius * radius] = dot_color
    return out


def filter_2(roi):
    return _dot_filter(roi, 6, (245, 245, 245), (15, 15, 15), 1.4)


def filter_3(roi):
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3] = (out[::3].astype(np.float32) * 0.72).astype(np.uint8)
    return out


def filter_5(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filter_6(roi):
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ], dtype=np.float32)
    sepia = cv2.transform(roi, sepia_kernel)
    vignette = _vignette(*roi.shape[:2])
    out = np.clip(sepia.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    # Keep a reusable noise buffer per resolution instead of allocating a new array repeatedly.
    h, w = roi.shape[:2]
    noise = _NOISE_CACHE.get((h, w))
    if noise is None:
        noise = np.empty((h, w, 3), dtype=np.uint8)
        _NOISE_CACHE[(h, w)] = noise
    cv2.randu(noise, 0, 25)
    return cv2.add(out, noise)


def filter_white(roi):
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    return cv2.addWeighted(blurred, 0.55, white, 0.45, 0)


def filter_pink(roi):
    return _dot_filter(roi, 5, (215, 190, 245), (55, 20, 130), 1.3)


def filter_grid(roi):
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (235, 235, 235), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (235, 235, 235), 1)
    return cv2.addWeighted(out, 0.75, roi, 0.25, 0)


def filter_fire(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def filter_neon(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out = np.zeros_like(roi)
    out[edges > 0] = (0, 255, 0)
    return out


def filter_pixel(roi):
    h, w = roi.shape[:2]
    sw, sh = max(1, w // 10), max(1, h // 10)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_invert(roi):
    return cv2.bitwise_not(roi)


def filter_popart(roi):
    h, w = roi.shape[:2]
    half_h, half_w = max(1, h // 2), max(1, w // 2)
    small = cv2.resize(roi, (half_w, half_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out = np.empty_like(roi)
    colors = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_WINTER, cv2.COLORMAP_SPRING]
    quads = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
    for cmap, (y0, x0) in zip(colors, quads):
        colored = cv2.applyColorMap(gray, cmap)
        out[y0:y0 + half_h, x0:x0 + half_w] = colored
    return out


FILTERS = [
    filter_grid,
    filter_1,
    filter_2,
    filter_3,
    filter_5,
    filter_6,
    filter_white,
    filter_pink,
    filter_fire,
    filter_neon,
    filter_pixel,
    filter_invert,
    filter_popart,
]

FILTER_NAMES = [
    "Grid", "Color Bands", "Halftone", "RGB Shift", "Jet", "Sepia", "Soft White",
    "Pink Halftone", "Inferno", "Neon", "Pixel", "Invert", "Pop Art",
]
