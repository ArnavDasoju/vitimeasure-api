"""
Vitiligo image analyser — pure OpenCV pipeline.

Pipeline:
  1. Detect skin region using YCrCb colour-space thresholds
  2. Apply CLAHE to the L channel (LAB) for contrast normalisation
  3. Detect depigmented (vitiligo) patches within the skin mask
  4. Find contours, compute bounding boxes and affected area percentage
  5. Draw cyan contour outlines on an annotated copy of the image
  6. Estimate dominant skin tone from the unaffected skin
"""

import base64
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class AnalysisResult:
    affected_percent: float
    unaffected_percent: float
    patch_count: int
    skin_tone: str
    ratio: str
    bounding_boxes: list[dict] = field(default_factory=list)
    annotated_image_b64: str = ""
    detection_confidence: float = 0.0


# ─── Skin-tone labels ────────────────────────────────────────────────────────

_SKIN_TONE_RANGES = [
    ("Very Light",  (200, 255)),
    ("Light",       (170, 200)),
    ("Medium Light", (140, 170)),
    ("Medium",      (110, 140)),
    ("Medium Dark", (80, 110)),
    ("Dark",        (50, 80)),
    ("Very Dark",   (0, 50)),
]


def _classify_skin_tone(mean_l: float) -> str:
    for label, (lo, hi) in _SKIN_TONE_RANGES:
        if lo <= mean_l < hi:
            return label
    return "Medium"


# ─── Core pipeline ───────────────────────────────────────────────────────────

def analyse_image(image_bytes: bytes) -> AnalysisResult:
    """Run the full vitiligo analysis pipeline on raw image bytes."""

    # Decode image
    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    # 1. Skin mask via YCrCb
    skin_mask = _detect_skin(img)
    skin_pixels = int(cv2.countNonZero(skin_mask))

    if skin_pixels < (h * w * 0.05):
        # Less than 5% skin detected — likely a bad photo
        return AnalysisResult(
            affected_percent=0.0,
            unaffected_percent=100.0,
            patch_count=0,
            skin_tone="Unknown",
            ratio="0:100",
            detection_confidence=0.1,
        )

    # 2. CLAHE on LAB L-channel for contrast normalisation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # 3. Detect depigmented areas within skin
    depig_mask = _detect_depigmentation(l_enhanced, skin_mask, img)
    depig_pixels = int(cv2.countNonZero(depig_mask))

    # 4. Find contours
    contours, _ = cv2.findContours(depig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out tiny contours (noise) — must be at least 0.1% of skin area
    min_area = max(skin_pixels * 0.001, 50)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Recalculate depig_pixels from valid contours only
    clean_mask = np.zeros_like(depig_mask)
    cv2.drawContours(clean_mask, contours, -1, 255, cv2.FILLED)
    depig_pixels = int(cv2.countNonZero(clean_mask))

    # 5. Metrics
    affected_pct = round((depig_pixels / skin_pixels) * 100, 2) if skin_pixels > 0 else 0.0
    unaffected_pct = round(100.0 - affected_pct, 2)
    patch_count = len(contours)

    # 6. Bounding boxes (normalised 0–1)
    bounding_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        bounding_boxes.append({
            "left": round(x / w, 4),
            "top": round(y / h, 4),
            "width": round(bw / w, 4),
            "height": round(bh / h, 4),
        })

    # 7. Skin tone from unaffected skin
    unaffected_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(clean_mask))
    if cv2.countNonZero(unaffected_mask) > 100:
        mean_l = float(cv2.mean(l_channel, mask=unaffected_mask)[0])
    else:
        mean_l = float(cv2.mean(l_channel, mask=skin_mask)[0])
    skin_tone = _classify_skin_tone(mean_l)

    # 8. Annotated image — cyan contour outlines
    annotated = img.copy()
    cv2.drawContours(annotated, contours, -1, (255, 255, 0), 2)  # cyan in BGR
    for bb in bounding_boxes:
        x1 = int(bb["left"] * w)
        y1 = int(bb["top"] * h)
        x2 = x1 + int(bb["width"] * w)
        y2 = y1 + int(bb["height"] * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)
    annotated_b64 = _encode_jpeg_b64(annotated)

    # 9. Detection confidence — heuristic based on skin coverage and contrast
    skin_coverage = skin_pixels / (h * w)
    confidence = min(1.0, skin_coverage * 1.5)
    if patch_count > 0:
        # Boost if contours have reasonable contrast
        contrast = _mean_contrast(l_enhanced, clean_mask, skin_mask)
        confidence = min(1.0, confidence * 0.7 + contrast * 0.3)
    confidence = round(confidence, 3)

    ratio = f"{round(affected_pct)}:{round(unaffected_pct)}"

    return AnalysisResult(
        affected_percent=affected_pct,
        unaffected_percent=unaffected_pct,
        patch_count=patch_count,
        skin_tone=skin_tone,
        ratio=ratio,
        bounding_boxes=bounding_boxes,
        annotated_image_b64=annotated_b64,
        detection_confidence=confidence,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to a BGR OpenCV array."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Cap resolution at 2000px on the long edge to keep processing fast
    max_dim = max(pil_img.size)
    if max_dim > 2000:
        scale = 2000 / max_dim
        pil_img = pil_img.resize(
            (int(pil_img.width * scale), int(pil_img.height * scale)),
            Image.LANCZOS,
        )
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _detect_skin(img: np.ndarray) -> np.ndarray:
    """Create a binary mask of skin pixels using YCrCb thresholds."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Empirical skin-colour bounds (works across a wide range of skin tones)
    lower = np.array([60, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Clean up with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _detect_depigmentation(
    l_enhanced: np.ndarray,
    skin_mask: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    """
    Detect depigmented (lighter) patches within the skin region.

    Uses adaptive thresholding on the CLAHE-enhanced L channel, then intersects
    with the skin mask. Also checks the A/B channels in LAB — vitiligo patches
    tend to be closer to neutral (lower chroma).
    """
    h, w = l_enhanced.shape

    # Adaptive threshold to find bright regions
    # Block size must be odd and large enough to capture patch context
    block_size = max(31, (min(h, w) // 8) | 1)  # ensure odd
    thresh = cv2.adaptiveThreshold(
        l_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, -10,
    )

    # Also do a global Otsu threshold as a second opinion
    mean_l = cv2.mean(l_enhanced, mask=skin_mask)[0]
    otsu_offset = max(0, mean_l * 0.15)
    _, global_thresh = cv2.threshold(
        l_enhanced, int(mean_l + otsu_offset), 255, cv2.THRESH_BINARY,
    )

    # Combine: pixel must be bright in BOTH methods
    combined = cv2.bitwise_and(thresh, global_thresh)

    # Restrict to skin region
    depig = cv2.bitwise_and(combined, skin_mask)

    # Refine with LAB chroma — vitiligo patches have low saturation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a_ch, b_ch = cv2.split(lab)
    # Neutral zone: A and B channels close to 128
    a_neutral = cv2.inRange(a_ch, 115, 145)
    b_neutral = cv2.inRange(b_ch, 110, 145)
    chroma_mask = cv2.bitwise_and(a_neutral, b_neutral)
    depig = cv2.bitwise_and(depig, chroma_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depig = cv2.morphologyEx(depig, cv2.MORPH_CLOSE, kernel, iterations=2)
    depig = cv2.morphologyEx(depig, cv2.MORPH_OPEN, kernel, iterations=1)

    return depig


def _mean_contrast(l_channel: np.ndarray, depig_mask: np.ndarray, skin_mask: np.ndarray) -> float:
    """Contrast ratio between depigmented and normal skin (0–1 scale)."""
    normal_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(depig_mask))
    if cv2.countNonZero(depig_mask) < 10 or cv2.countNonZero(normal_mask) < 10:
        return 0.5
    depig_mean = cv2.mean(l_channel, mask=depig_mask)[0]
    normal_mean = cv2.mean(l_channel, mask=normal_mask)[0]
    diff = abs(depig_mean - normal_mean) / 255.0
    return min(1.0, diff * 3)  # scale up — a 30+ L* difference is strong


def _encode_jpeg_b64(img: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR image as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
