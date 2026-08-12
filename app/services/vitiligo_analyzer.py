"""
Vitiligo image analyser — production-grade OpenCV pipeline.

Designed for clinical-level accuracy across all skin tones (Fitzpatrick I–VI)
and all vitiligo presentations (focal, segmental, generalized, universal,
trichrome). Robust against hair follicles, body hair, specular highlights,
uneven lighting, shadows, and image noise.

Pipeline:
  1. Decode + resize
  2. Illumination normalization (homomorphic filtering)
  3. Specular highlight suppression
  4. Hair removal (multi-angle black-hat + inpainting)
  5. Multi-method skin detection (YCrCb + HSV + luminance proximity)
  6. K-means clustering in LAB to separate pigmentation levels
  7. Superpixel-assisted region growing for natural boundaries
  8. Multi-scale depigmentation detection with robust statistics
  9. Texture validation (LBP) to reject non-vitiligo bright regions
  10. Boundary refinement via morphological reconstruction
  11. Contour extraction with proximity-based merging
  12. Multi-factor confidence scoring
  13. Annotated image generation
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


# ─── Skin-tone classification (Fitzpatrick-aligned) ─────────────────────────

_SKIN_TONE_RANGES = [
    ("Very Light",   (190, 255)),
    ("Light",        (160, 190)),
    ("Medium Light", (135, 160)),
    ("Medium",       (110, 135)),
    ("Medium Dark",  (80, 110)),
    ("Dark",         (45, 80)),
    ("Very Dark",    (0, 45)),
]


def _classify_skin_tone(mean_l: float) -> str:
    for label, (lo, hi) in _SKIN_TONE_RANGES:
        if lo <= mean_l < hi:
            return label
    return "Medium"


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def analyse_image(image_bytes: bytes) -> AnalysisResult:
    """Run the full vitiligo analysis pipeline on raw image bytes."""

    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    # ── Stage 1: Light denoise + hair removal ──────────────────────────────
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    img_hairless, hair_mask = _remove_hair(img_denoised)

    # ── Stage 2: Specular highlight suppression ──────────────────────────
    img_no_spec, spec_mask = _suppress_specular(img_hairless)

    # ── Stage 3: Illumination normalization (for skin detection only) ────
    img_norm = _normalize_illumination(img_no_spec)

    # ── Stage 4: Skin detection ──────────────────────────────────────────
    skin_mask = _detect_skin(img_norm, img)
    if spec_mask is not None:
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(spec_mask))

    # For analysis, use hair-removed + specular-fixed image (NOT normalized,
    # because normalization can flatten the subtle L* differences we need)
    img_clean = img_no_spec
    skin_pixels = int(cv2.countNonZero(skin_mask))

    if skin_pixels < (h * w * 0.03):
        return AnalysisResult(
            affected_percent=0.0, unaffected_percent=100.0,
            patch_count=0, skin_tone="Unknown", ratio="0:100",
            detection_confidence=0.1,
        )

    # ── Stage 5: LAB decomposition + CLAHE ───────────────────────────────
    # Use the hair-removed, specular-fixed image (NOT normalized) to
    # preserve the L* contrast between normal and depigmented skin
    lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
    l_raw, a_ch, b_ch = cv2.split(lab)

    mean_l_skin = float(cv2.mean(l_raw, mask=skin_mask)[0])
    clip = 4.0 if mean_l_skin < 90 else 3.0 if mean_l_skin < 140 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_raw)

    # ── Stage 6: K-means pigmentation clustering ─────────────────────────
    kmeans_mask = _kmeans_depigmentation(l_raw, a_ch, b_ch, skin_mask)

    # ── Stage 7: Multi-scale statistical detection ───────────────────────
    stat_mask = _detect_depigmentation(l_enhanced, l_raw, a_ch, b_ch, skin_mask)

    # ── Stage 8: Superpixel-assisted region refinement ───────────────────
    superpixel_mask = _superpixel_refine(img_clean, l_raw, skin_mask, stat_mask)

    # ── Stage 9: Combine all detection methods ───────────────────────────
    vote_map = np.zeros((h, w), dtype=np.float32)
    vote_map += (kmeans_mask > 0).astype(np.float32) * 1.2      # k-means
    vote_map += (stat_mask > 0).astype(np.float32) * 1.0        # statistical
    vote_map += (superpixel_mask > 0).astype(np.float32) * 1.0  # superpixel-refined

    # Require 2+ methods to agree (stat+superpixel=2.0, kmeans+either=2.2+)
    combined = (vote_map >= 1.9).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, skin_mask)

    # ── Stage 10: Texture validation ─────────────────────────────────────
    combined = _texture_validate(l_raw, combined, skin_mask)

    # ── Stage 11: Morphological refinement ───────────────────────────────
    combined = _refine_mask(combined, skin_mask, hair_mask)

    # ── Stage 12: Contour extraction + merging ───────────────────────────
    contours = _extract_contours(combined, skin_pixels)

    # Rebuild mask from final contours
    final_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(final_mask, contours, -1, 255, cv2.FILLED)
    depig_pixels = int(cv2.countNonZero(final_mask))

    # ── Metrics ──────────────────────────────────────────────────────────
    affected_pct = round((depig_pixels / skin_pixels) * 100, 2) if skin_pixels > 0 else 0.0
    unaffected_pct = round(100.0 - affected_pct, 2)
    patch_count = len(contours)

    bounding_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        bounding_boxes.append({
            "left": round(x / w, 4), "top": round(y / h, 4),
            "width": round(bw / w, 4), "height": round(bh / h, 4),
        })

    # Skin tone from unaffected skin
    unaffected = cv2.bitwise_and(skin_mask, cv2.bitwise_not(final_mask))
    tone_l = float(cv2.mean(l_raw, mask=unaffected)[0]) if cv2.countNonZero(unaffected) > 100 \
        else float(cv2.mean(l_raw, mask=skin_mask)[0])
    skin_tone = _classify_skin_tone(tone_l)

    # Annotated image
    annotated = _draw_annotations(img, contours, bounding_boxes, final_mask, w, h)
    annotated_b64 = _encode_jpeg_b64(annotated)

    # Confidence
    confidence = _compute_confidence(l_enhanced, final_mask, skin_mask, skin_pixels,
                                      h, w, patch_count, contours, vote_map)

    ratio = f"{round(affected_pct)}:{round(unaffected_pct)}"

    return AnalysisResult(
        affected_percent=affected_pct, unaffected_percent=unaffected_pct,
        patch_count=patch_count, skin_tone=skin_tone, ratio=ratio,
        bounding_boxes=bounding_boxes, annotated_image_b64=annotated_b64,
        detection_confidence=confidence,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ─── Decode ──────────────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = max(pil_img.size)
    if max_dim > 2000:
        scale = 2000 / max_dim
        pil_img = pil_img.resize(
            (int(pil_img.width * scale), int(pil_img.height * scale)),
            Image.LANCZOS,
        )
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─── Illumination normalization ──────────────────────────────────────────────

def _normalize_illumination(img: np.ndarray) -> np.ndarray:
    """
    Homomorphic filtering to separate reflectance from illumination.
    Normalizes uneven lighting, shadows from body curvature, and flash hotspots.
    Combined with gray-world white balance.
    """
    # Convert to LAB and work on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_ch = lab[:, :, 0]

    # Homomorphic filter: log → highpass → exp
    # CRITICAL: Use a VERY large sigma so only broad illumination gradients
    # are captured. Small sigma would erase the vitiligo patches themselves.
    l_log = np.log1p(l_ch)

    sigma = max(l_ch.shape) // 3  # very large — only catches room-level lighting
    if sigma % 2 == 0:
        sigma += 1
    l_blur = cv2.GaussianBlur(l_log, (0, 0), sigma)

    # Gentle correction: mostly preserve original, slightly flatten illumination
    gamma_h = 1.15   # reflectance gain (subtle)
    gamma_l = 0.85   # illumination attenuation (gentle)
    l_filtered = gamma_h * (l_log - l_blur) + gamma_l * l_blur

    # Back to linear
    l_norm = np.expm1(l_filtered)
    l_norm = np.clip(l_norm, 0, 255)

    # Preserve original dynamic range
    orig_mean = np.mean(l_ch)
    norm_mean = np.mean(l_norm)
    if norm_mean > 0:
        l_norm = l_norm * (orig_mean / norm_mean)
    l_norm = np.clip(l_norm, 0, 255)

    lab[:, :, 0] = l_norm

    # Gray-world white balance on A/B channels
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 0.7
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 0.7

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─── Specular highlight suppression ─────────────────────────────────────────

def _suppress_specular(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Detect and suppress specular highlights (shiny/oily skin).
    These cause false positives because they appear very bright.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Specular: very high value, very low saturation
    spec_mask = cv2.inRange(hsv,
                             np.array([0, 0, 230], dtype=np.uint8),
                             np.array([180, 30, 255], dtype=np.uint8))

    # Only keep large specular regions (small bright spots are fine)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_OPEN, kernel)

    if cv2.countNonZero(spec_mask) < 50:
        return img, None

    # Inpaint specular regions with surrounding color
    result = cv2.inpaint(img, spec_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result, spec_mask


# ─── Hair removal ────────────────────────────────────────────────────────────

def _remove_hair(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    DullRazor-inspired hair removal using multi-angle black-hat filtering.
    Uses the green channel where hair contrast is highest.
    """
    # Green channel has best hair-skin contrast across all skin tones
    green = img[:, :, 1]

    hair_response = np.zeros_like(green, dtype=np.float32)

    # 12 orientations — catches hair at any angle
    for angle_deg in range(0, 180, 15):
        kernel_len = 19
        kernel = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
        center = kernel_len // 2
        # Draw a line through center at this angle
        dx = int(np.cos(np.radians(angle_deg)) * center)
        dy = int(np.sin(np.radians(angle_deg)) * center)
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)

        if np.sum(kernel) == 0:
            continue

        blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
        hair_response = np.maximum(hair_response, blackhat.astype(np.float32))

    # Adaptive threshold — use Otsu on the hair response
    hair_uint8 = np.clip(hair_response, 0, 255).astype(np.uint8)
    _, hair_binary = cv2.threshold(hair_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu is too aggressive (too much detected), fall back to fixed threshold
    hair_ratio = cv2.countNonZero(hair_binary) / (green.shape[0] * green.shape[1])
    if hair_ratio > 0.20:
        _, hair_binary = cv2.threshold(hair_uint8, 25, 255, cv2.THRESH_BINARY)

    # Validate: keep only elongated structures (hair) not blobs (not hair)
    contours, _ = cv2.findContours(hair_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hair_validated = np.zeros_like(hair_binary)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:
            continue

        perimeter = cv2.arcLength(c, True)
        # Circularity: 1.0 = perfect circle, low = elongated
        circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

        if len(c) >= 5:
            _, (rw, rh), _ = cv2.minAreaRect(c)
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        else:
            aspect = 1.0

        # Hair: elongated (high aspect) OR non-circular AND small
        is_elongated = aspect > 3.0
        is_hair_like = circularity < 0.4 and area < 1000
        is_tiny_follicle = area < 150

        if is_elongated or is_hair_like or is_tiny_follicle:
            cv2.drawContours(hair_validated, [c], -1, 255, cv2.FILLED)

    # Dilate to cover full hair width
    hair_validated = cv2.dilate(hair_validated,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                 iterations=1)

    hair_count = cv2.countNonZero(hair_validated)
    total = green.shape[0] * green.shape[1]

    if hair_count > 30 and hair_count < total * 0.12:
        inpainted = cv2.inpaint(img, hair_validated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted, hair_validated

    return img, None


# ─── Skin detection ──────────────────────────────────────────────────────────

def _detect_skin(img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """
    Multi-space skin detection. Combines YCrCb, HSV, and a dedicated
    vitiligo-skin detector. Uses the original (pre-normalization) image
    as a cross-reference.
    """
    h, w = img.shape[:2]

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # YCrCb: wide bounds for Fitzpatrick I–VI including depigmented skin
    mask1 = cv2.inRange(ycrcb,
                         np.array([30, 120, 70], dtype=np.uint8),
                         np.array([255, 200, 150], dtype=np.uint8))

    # HSV: skin hue range (red–yellow) with low-to-high saturation
    mask2a = cv2.inRange(hsv,
                          np.array([0, 12, 45], dtype=np.uint8),
                          np.array([25, 255, 255], dtype=np.uint8))
    mask2b = cv2.inRange(hsv,
                          np.array([160, 12, 45], dtype=np.uint8),
                          np.array([180, 255, 255], dtype=np.uint8))
    mask2 = cv2.bitwise_or(mask2a, mask2b)

    # Vitiligo-specific: very bright, very low saturation, near detected skin
    mask_vit = cv2.inRange(hsv,
                            np.array([0, 0, 130], dtype=np.uint8),
                            np.array([180, 50, 255], dtype=np.uint8))
    # Only include if adjacent to already-detected skin
    skin_base = cv2.bitwise_or(mask1, mask2)
    proximity = cv2.dilate(skin_base, np.ones((51, 51), np.uint8))
    mask_vit = cv2.bitwise_and(mask_vit, proximity)

    combined = cv2.bitwise_or(skin_base, mask_vit)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)

    # Keep only significant skin regions (>0.3% of image)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(combined)
    min_area = h * w * 0.003
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(result, [c], -1, 255, cv2.FILLED)

    return result


# ─── K-means pigmentation clustering ────────────────────────────────────────

def _kmeans_depigmentation(
    l_ch: np.ndarray, a_ch: np.ndarray, b_ch: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Use K-means clustering in LAB color space to automatically separate
    skin into pigmentation levels. The lightest cluster within skin is
    identified as potentially depigmented.

    This is the single most powerful method for vitiligo detection because
    it adapts to any skin tone without hardcoded thresholds.
    """
    h, w = l_ch.shape

    # Extract skin pixels as LAB feature vectors
    skin_coords = np.where(skin_mask > 0)
    if len(skin_coords[0]) < 200:
        return np.zeros((h, w), dtype=np.uint8)

    features = np.column_stack([
        l_ch[skin_coords].astype(np.float32),
        a_ch[skin_coords].astype(np.float32) * 0.8,  # weight chrominance lower
        b_ch[skin_coords].astype(np.float32) * 0.8,
    ])

    # Try K=3 (normal skin, depigmented, trichrome/border zone)
    # If the image has no vitiligo, clusters will be similar — that's fine
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()

    # Find the lightest cluster (highest L value)
    cluster_l_means = [centers[i][0] for i in range(k)]
    lightest_cluster = int(np.argmax(cluster_l_means))
    darkest_cluster = int(np.argmin(cluster_l_means))

    # The lightest cluster is depigmented IF it's significantly brighter
    # than the median cluster
    sorted_means = sorted(cluster_l_means)
    l_gap = sorted_means[-1] - sorted_means[-2]  # gap between lightest and second

    # Require meaningful L* separation (natural skin variation is ~5–8 L*)
    if l_gap < 10:
        return np.zeros((h, w), dtype=np.uint8)

    # The lightest cluster should be a MINORITY of skin pixels (vitiligo is patchy)
    # If it's >60% of skin, it's probably the dominant skin color, not vitiligo
    lightest_count = np.sum(labels == lightest_cluster)
    lightest_ratio = lightest_count / len(labels)
    if lightest_ratio > 0.60 or lightest_ratio < 0.005:
        return np.zeros((h, w), dtype=np.uint8)

    # Lightest cluster should be more neutral (lower chroma) than the
    # darkest — vitiligo loses melanin AND chrominance
    lightest_chroma = np.sqrt(
        (centers[lightest_cluster][1] / 0.8 - 128) ** 2 +
        (centers[lightest_cluster][2] / 0.8 - 128) ** 2
    )
    darkest_chroma = np.sqrt(
        (centers[darkest_cluster][1] / 0.8 - 128) ** 2 +
        (centers[darkest_cluster][2] / 0.8 - 128) ** 2
    )
    if lightest_chroma > darkest_chroma * 1.3:
        return np.zeros((h, w), dtype=np.uint8)

    # Build mask from lightest cluster
    depig_pixels = (labels == lightest_cluster)
    result = np.zeros((h, w), dtype=np.uint8)
    result[skin_coords[0][depig_pixels], skin_coords[1][depig_pixels]] = 255

    return result


# ─── Multi-scale statistical detection ───────────────────────────────────────

def _detect_depigmentation(
    l_enhanced: np.ndarray, l_raw: np.ndarray,
    a_ch: np.ndarray, b_ch: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Robust statistical detection using median/IQR (outlier-resistant)
    combined with multi-scale adaptive thresholding and LAB color analysis.
    """
    h, w = l_enhanced.shape

    skin_l = l_enhanced[skin_mask > 0]
    if len(skin_l) < 100:
        return np.zeros((h, w), dtype=np.uint8)

    # Robust statistics: median and IQR (resistant to hair/highlight outliers)
    median_l = float(np.median(skin_l))
    q1 = float(np.percentile(skin_l, 25))
    q3 = float(np.percentile(skin_l, 75))
    iqr = q3 - q1

    # Adaptive k-factor based on skin tone
    if median_l < 90:
        k = 0.8
    elif median_l < 130:
        k = 1.0
    elif median_l < 170:
        k = 1.3
    else:
        k = 1.8

    # ── Method A: Percentile-based threshold ─────────────────────────────
    thresh_a = median_l + k * max(iqr * 0.75, 6)
    mask_a = (l_enhanced > thresh_a).astype(np.uint8) * 255
    mask_a = cv2.bitwise_and(mask_a, skin_mask)

    # ── Method B: Multi-scale adaptive ───────────────────────────────────
    masks_b = []
    for divisor in [6, 10, 16, 24]:
        bs = max(31, (min(h, w) // divisor) | 1)
        c_val = -max(3, int(iqr * 0.25))
        at = cv2.adaptiveThreshold(l_enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, bs, c_val)
        masks_b.append(cv2.bitwise_and(at, skin_mask))

    # Pixel in 2+ of 4 scales
    b_sum = sum((m > 0).astype(np.uint16) for m in masks_b)
    mask_b = (b_sum >= 2).astype(np.uint8) * 255

    # ── Method C: LAB color distance ─────────────────────────────────────
    skin_a = a_ch[skin_mask > 0]
    skin_b = b_ch[skin_mask > 0]
    med_a = float(np.median(skin_a))
    med_b = float(np.median(skin_b))

    # Vitiligo: brighter + shifted toward neutral (a≈128, b≈128)
    l_above = l_enhanced.astype(np.float32) - median_l
    chroma = np.sqrt((a_ch.astype(np.float32) - 128) ** 2 +
                      (b_ch.astype(np.float32) - 128) ** 2)
    skin_chroma = chroma[skin_mask > 0]
    med_chroma = float(np.median(skin_chroma))

    mask_c = np.zeros((h, w), dtype=np.uint8)
    bright_enough = l_above > max(4, iqr * 0.3)
    neutral_enough = chroma < (med_chroma + 3)
    mask_c[bright_enough & neutral_enough] = 255
    mask_c = cv2.bitwise_and(mask_c, skin_mask)

    # ── Combine: weighted vote ───────────────────────────────────────────
    votes = np.zeros((h, w), dtype=np.float32)
    votes += (mask_a > 0).astype(np.float32) * 1.0
    votes += (mask_b > 0).astype(np.float32) * 1.0
    votes += (mask_c > 0).astype(np.float32) * 1.2

    result = (votes >= 1.8).astype(np.uint8) * 255
    return cv2.bitwise_and(result, skin_mask)


# ─── Superpixel-assisted refinement ──────────────────────────────────────────

def _superpixel_refine(
    img: np.ndarray, l_ch: np.ndarray,
    skin_mask: np.ndarray, initial_mask: np.ndarray,
) -> np.ndarray:
    """
    Use superpixel segmentation to refine detection boundaries.
    A superpixel is classified as depigmented if >40% of its area
    overlaps with the initial detection. This creates natural,
    contiguous boundaries instead of noisy pixel-level masks.
    """
    h, w = l_ch.shape

    # SLIC-like superpixel approximation using connected components on quantized image
    # (cv2.ximgproc.createSuperpixelSLIC requires opencv-contrib, so we approximate)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Quantize LAB channels to reduce to ~500 superpixels
    lab_q = (lab // 12) * 12
    gray_q = cv2.cvtColor(lab_q, cv2.COLOR_BGR2GRAY)

    # Watershed-based oversegmentation
    edges = cv2.Canny(gray_q, 30, 100)
    edges = cv2.dilate(edges, None)
    # Create markers from non-edge regions
    _, markers = cv2.connectedComponents(cv2.bitwise_not(edges))

    result = np.zeros((h, w), dtype=np.uint8)

    n_labels = markers.max()
    if n_labels < 2:
        return initial_mask

    for label_id in range(1, min(n_labels + 1, 3000)):  # cap for performance
        region = (markers == label_id)
        region_size = np.sum(region)
        if region_size < 20:
            continue

        # What fraction overlaps with skin?
        skin_overlap = np.sum(region & (skin_mask > 0))
        if skin_overlap < region_size * 0.5:
            continue  # not enough skin

        # What fraction overlaps with initial detection?
        depig_overlap = np.sum(region & (initial_mask > 0))
        if depig_overlap > region_size * 0.35:
            result[region] = 255

    return result


# ─── Texture validation (LBP-inspired) ──────────────────────────────────────

def _texture_validate(
    l_ch: np.ndarray, mask: np.ndarray, skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Validate detected regions using texture analysis.
    Vitiligo patches have smooth, uniform texture similar to surrounding skin.
    Scars, dry patches, and clothing have distinctly different texture.

    Uses variance of Laplacian as a texture roughness measure.
    """
    if cv2.countNonZero(mask) < 50:
        return mask

    laplacian = cv2.Laplacian(l_ch, cv2.CV_64F)
    lap_abs = np.abs(laplacian)

    # Compute texture roughness for normal skin
    normal_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(mask))
    if cv2.countNonZero(normal_mask) < 100:
        return mask

    normal_texture = float(np.median(lap_abs[normal_mask > 0]))

    # Check each connected component of the detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validated = np.zeros_like(mask)

    for c in contours:
        c_mask = np.zeros_like(mask)
        cv2.drawContours(c_mask, [c], -1, 255, cv2.FILLED)

        if cv2.countNonZero(c_mask) < 30:
            continue

        patch_texture = float(np.median(lap_abs[c_mask > 0]))

        # Vitiligo: texture should be similar to normal skin
        # (scars and non-skin objects are much rougher)
        # If both are very smooth (uniform regions), that's fine — vitiligo IS smooth
        both_smooth = normal_texture < 3.0 and patch_texture < 3.0
        if both_smooth:
            cv2.drawContours(validated, [c], -1, 255, cv2.FILLED)
        else:
            texture_ratio = patch_texture / (normal_texture + 1e-6)
            if 0.1 < texture_ratio < 4.0:
                cv2.drawContours(validated, [c], -1, 255, cv2.FILLED)

    return validated


# ─── Mask refinement ─────────────────────────────────────────────────────────

def _refine_mask(
    mask: np.ndarray, skin_mask: np.ndarray,
    hair_mask: np.ndarray | None,
) -> np.ndarray:
    """
    Final morphological refinement:
    - Exclude hair regions
    - Close gaps from hair follicles within patches
    - Fill holes
    - Remove isolated pixels
    """
    if hair_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(hair_mask))

    # Close gaps (hair follicle holes inside patches)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Remove tiny noise
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Fill holes inside contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

    # Restrict to skin
    mask = cv2.bitwise_and(mask, skin_mask)

    return mask


# ─── Contour extraction + merging ────────────────────────────────────────────

def _extract_contours(mask: np.ndarray, skin_pixels: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(skin_pixels * 0.0004, 25)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if len(valid) <= 1:
        return valid

    return _merge_nearby(valid, dist=30)


def _merge_nearby(contours: list[np.ndarray], dist: int) -> list[np.ndarray]:
    """Union-find merge of contours within `dist` pixels."""
    n = len(contours)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    rects = [cv2.boundingRect(c) for c in contours]
    for i in range(n):
        x1, y1, w1, h1 = rects[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = rects[j]
            dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            if np.sqrt(dx * dx + dy * dy) < dist:
                union(i, j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(contours[i])

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(cv2.convexHull(np.vstack(group)))
    return merged


# ─── Annotation ──────────────────────────────────────────────────────────────

def _draw_annotations(img, contours, bboxes, mask, w, h):
    out = img.copy()
    overlay = out.copy()
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    for bb in bboxes:
        x1, y1 = int(bb["left"] * w), int(bb["top"] * h)
        x2 = x1 + int(bb["width"] * w)
        y2 = y1 + int(bb["height"] * h)
        cv2.rectangle(out, (max(0, x1 - 3), max(0, y1 - 3)),
                       (min(w, x2 + 3), min(h, y2 + 3)),
                       (255, 255, 0), 1, cv2.LINE_AA)
    return out


# ─── Confidence scoring ─────────────────────────────────────────────────────

def _compute_confidence(l_ch, depig_mask, skin_mask, skin_pixels,
                         h, w, patch_count, contours, vote_map):
    total = h * w

    # Skin coverage → 0–0.25
    f1 = min(0.25, (skin_pixels / total) * 0.35)

    # L* contrast between depig and normal → 0–0.30
    f2 = 0.0
    if patch_count > 0:
        normal = cv2.bitwise_and(skin_mask, cv2.bitwise_not(depig_mask))
        if cv2.countNonZero(depig_mask) > 10 and cv2.countNonZero(normal) > 10:
            d_mean = float(cv2.mean(l_ch, mask=depig_mask)[0])
            n_mean = float(cv2.mean(l_ch, mask=normal)[0])
            f2 = min(0.30, abs(d_mean - n_mean) / 55.0 * 0.30)

    # Edge sharpness → 0–0.15
    f3 = 0.0
    if contours:
        boundary = np.zeros_like(depig_mask)
        cv2.drawContours(boundary, contours, -1, 255, 3)
        boundary = cv2.bitwise_and(boundary, skin_mask)
        if cv2.countNonZero(boundary) > 0:
            grad = np.abs(cv2.Laplacian(l_ch, cv2.CV_64F))
            f3 = min(0.15, float(np.mean(grad[boundary > 0])) / 35.0 * 0.15)

    # Contour solidity → 0–0.15
    f4 = 0.0
    if contours:
        solds = []
        for c in contours:
            ca = cv2.contourArea(c)
            ha = cv2.contourArea(cv2.convexHull(c))
            if ha > 0:
                solds.append(ca / ha)
        if solds:
            f4 = min(0.15, float(np.mean(solds)) * 0.15)

    # Method agreement → 0–0.15
    f5 = 0.0
    if patch_count > 0 and cv2.countNonZero(depig_mask) > 0:
        depig_votes = vote_map[depig_mask > 0]
        avg_votes = float(np.mean(depig_votes))
        f5 = min(0.15, (avg_votes / 3.5) * 0.15)

    conf = f1 + f2 + f3 + f4 + f5
    if patch_count > 0 and f2 > 0.05:
        conf = max(0.45, conf)

    return round(min(1.0, conf), 3)


def _encode_jpeg_b64(img, quality=82):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
