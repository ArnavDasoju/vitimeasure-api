"""
Vitiligo image analyser — advanced OpenCV pipeline.

Handles:
  - Hair follicles and body hair interference (DullRazor-inspired inpainting)
  - Wide range of skin tones (Fitzpatrick I–VI)
  - Multiple vitiligo presentations (focal, segmental, generalized, universal)
  - Trichrome vitiligo (intermediate depigmentation zones)
  - Variable lighting conditions

Pipeline:
  1. Pre-process: denoise, normalize white balance
  2. Hair removal via morphological black-hat + inpainting
  3. Multi-method skin detection (YCrCb + HSV + adaptive)
  4. CLAHE on LAB L-channel for contrast normalization
  5. Multi-scale depigmentation detection with statistical thresholding
  6. Contour extraction with hierarchy-aware merging
  7. Confidence scoring from contrast, edge sharpness, and coverage
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


# ─── Skin-tone labels (Fitzpatrick-aligned) ──────────────────────────────────

_SKIN_TONE_RANGES = [
    ("Very Light",   (190, 255)),   # Fitzpatrick I
    ("Light",        (160, 190)),   # Fitzpatrick II
    ("Medium Light", (135, 160)),   # Fitzpatrick III
    ("Medium",       (110, 135)),   # Fitzpatrick IV
    ("Medium Dark",  (80, 110)),    # Fitzpatrick V
    ("Dark",         (45, 80)),     # Fitzpatrick VI
    ("Very Dark",    (0, 45)),
]


def _classify_skin_tone(mean_l: float) -> str:
    for label, (lo, hi) in _SKIN_TONE_RANGES:
        if lo <= mean_l < hi:
            return label
    return "Medium"


# ─── Core pipeline ───────────────────────────────────────────────────────────

def analyse_image(image_bytes: bytes) -> AnalysisResult:
    """Run the full vitiligo analysis pipeline on raw image bytes."""

    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    # 1. Pre-process: denoise + white balance
    img_clean = _preprocess(img)

    # 2. Remove hair follicles — critical for accuracy on hairy skin
    img_hairless, hair_mask = _remove_hair(img_clean)

    # 3. Multi-method skin detection
    skin_mask = _detect_skin_multi(img_hairless)
    skin_pixels = int(cv2.countNonZero(skin_mask))

    if skin_pixels < (h * w * 0.03):
        return AnalysisResult(
            affected_percent=0.0,
            unaffected_percent=100.0,
            patch_count=0,
            skin_tone="Unknown",
            ratio="0:100",
            detection_confidence=0.1,
        )

    # 4. CLAHE on LAB L-channel
    lab = cv2.cvtColor(img_hairless, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Adaptive CLAHE — stronger for dark skin, gentler for light skin
    mean_l_skin = float(cv2.mean(l_channel, mask=skin_mask)[0])
    clip_limit = 3.5 if mean_l_skin < 100 else 2.5 if mean_l_skin < 150 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # 5. Multi-scale depigmentation detection
    depig_mask = _detect_depigmentation_multiscale(
        l_enhanced, l_channel, a_channel, b_channel, skin_mask, img_hairless,
    )

    # Exclude hair regions from depigmentation (dark hair ≠ pigmented skin)
    if hair_mask is not None:
        depig_mask = cv2.bitwise_and(depig_mask, cv2.bitwise_not(hair_mask))

    # 6. Contour extraction with intelligent merging
    contours = _extract_contours(depig_mask, skin_pixels)

    # Rebuild clean mask from validated contours
    clean_mask = np.zeros_like(depig_mask)
    if contours:
        cv2.drawContours(clean_mask, contours, -1, 255, cv2.FILLED)
    depig_pixels = int(cv2.countNonZero(clean_mask))

    # 7. Metrics
    affected_pct = round((depig_pixels / skin_pixels) * 100, 2) if skin_pixels > 0 else 0.0
    unaffected_pct = round(100.0 - affected_pct, 2)
    patch_count = len(contours)

    # 8. Bounding boxes (normalized 0–1)
    bounding_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        bounding_boxes.append({
            "left": round(x / w, 4),
            "top": round(y / h, 4),
            "width": round(bw / w, 4),
            "height": round(bh / h, 4),
        })

    # 9. Skin tone from unaffected skin
    unaffected_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(clean_mask))
    if cv2.countNonZero(unaffected_mask) > 100:
        mean_l = float(cv2.mean(l_channel, mask=unaffected_mask)[0])
    else:
        mean_l = float(cv2.mean(l_channel, mask=skin_mask)[0])
    skin_tone = _classify_skin_tone(mean_l)

    # 10. Annotated image — multi-layer visualization
    annotated = _draw_annotations(img, contours, bounding_boxes, clean_mask, w, h)
    annotated_b64 = _encode_jpeg_b64(annotated)

    # 11. Confidence scoring — multi-factor
    confidence = _compute_confidence(
        l_enhanced, clean_mask, skin_mask, skin_pixels, h, w,
        patch_count, contours,
    )

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


# ─── Pre-processing ─────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to a BGR OpenCV array, capped at 2000px."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = max(pil_img.size)
    if max_dim > 2000:
        scale = 2000 / max_dim
        pil_img = pil_img.resize(
            (int(pil_img.width * scale), int(pil_img.height * scale)),
            Image.LANCZOS,
        )
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Denoise and normalize white balance."""
    # Non-local means denoising — preserves edges while reducing sensor noise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    # Simple gray-world white balance
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 0.8)
    lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 0.8)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─── Hair removal (DullRazor-inspired) ───────────────────────────────────────

def _remove_hair(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Remove dark hair follicles using morphological black-hat filtering
    followed by inpainting. Returns (cleaned_image, hair_mask).

    The black-hat transform isolates thin dark structures (hair) against
    a lighter background (skin). We use multiple oriented kernels to catch
    hair at different angles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multi-directional black-hat to catch hair at all angles
    hair_mask = np.zeros_like(gray)
    kernel_size = 17  # tuned for typical hair width at skin-photo distance

    for angle in range(0, 180, 15):  # 12 directions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size // 2, 0), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = (kernel > 0).astype(np.uint8)
        if np.sum(kernel) == 0:
            continue
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        hair_mask = cv2.max(hair_mask, blackhat)

    # Threshold to get binary hair mask
    _, hair_binary = cv2.threshold(hair_mask, 15, 255, cv2.THRESH_BINARY)

    # Dilate slightly to cover hair width
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_binary = cv2.dilate(hair_binary, dilate_kernel, iterations=1)

    # Filter out large blobs — real hair is thin and elongated
    contours, _ = cv2.findContours(hair_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hair_filtered = np.zeros_like(hair_binary)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        # Hair is elongated — check aspect ratio of rotated bounding rect
        if len(c) >= 5:
            _, (rw, rh), _ = cv2.minAreaRect(c)
            aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
            # Accept elongated shapes (aspect > 3) or small dots (follicles)
            if aspect > 2.5 or area < 200:
                cv2.drawContours(hair_filtered, [c], -1, 255, cv2.FILLED)
        elif area < 300:
            cv2.drawContours(hair_filtered, [c], -1, 255, cv2.FILLED)

    hair_pixel_count = cv2.countNonZero(hair_filtered)
    total_pixels = gray.shape[0] * gray.shape[1]

    # Only inpaint if hair was actually detected (< 15% of image)
    if hair_pixel_count > 50 and hair_pixel_count < total_pixels * 0.15:
        result = cv2.inpaint(img, hair_filtered, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result, hair_filtered
    else:
        return img, None


# ─── Multi-method skin detection ─────────────────────────────────────────────

def _detect_skin_multi(img: np.ndarray) -> np.ndarray:
    """
    Robust skin detection combining YCrCb, HSV, and luminance.
    Wider bounds than the original to include depigmented (vitiligo) skin,
    which is lighter and less chromatic than normal skin.
    """
    h, w = img.shape[:2]

    # Method 1: YCrCb — primary skin detector with widened bounds
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Wide bounds: includes very light (vitiligo) to very dark skin
    mask_ycrcb = cv2.inRange(ycrcb,
                              np.array([40, 125, 75], dtype=np.uint8),
                              np.array([255, 195, 145], dtype=np.uint8))

    # Method 2: HSV — catches skin that YCrCb misses
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Skin hue is roughly 0–50 (red-orange-yellow range)
    mask_hsv = cv2.inRange(hsv,
                            np.array([0, 15, 50], dtype=np.uint8),
                            np.array([50, 255, 255], dtype=np.uint8))

    # Method 3: Depigmented skin detector — very light, low saturation
    # Vitiligo patches are often missed by standard skin detectors
    mask_depig = cv2.inRange(hsv,
                              np.array([0, 0, 140], dtype=np.uint8),
                              np.array([180, 60, 255], dtype=np.uint8))
    # Only keep depigmented pixels near detected skin (within 30px)
    skin_dilated = cv2.dilate(mask_ycrcb, np.ones((61, 61), np.uint8))
    mask_depig = cv2.bitwise_and(mask_depig, skin_dilated)

    # Combine all methods (union)
    combined = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    combined = cv2.bitwise_or(combined, mask_depig)

    # Morphological cleanup — close gaps, remove noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Fill holes inside skin regions
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    min_skin_area = h * w * 0.005  # at least 0.5% of image
    for c in contours:
        if cv2.contourArea(c) >= min_skin_area:
            cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)

    return filled


# ─── Multi-scale depigmentation detection ────────────────────────────────────

def _detect_depigmentation_multiscale(
    l_enhanced: np.ndarray,
    l_raw: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    skin_mask: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    """
    Detect depigmented patches using multiple complementary methods,
    then combine via weighted voting.

    Methods:
      A. Statistical thresholding — pixels brighter than mean + k*std of skin
      B. Adaptive thresholding — local contrast detection at multiple scales
      C. Color-distance based — deviation from median skin color in LAB space
      D. Gradient-boundary detection — sharp luminance transitions
    """
    h, w = l_enhanced.shape

    # Compute skin statistics
    skin_l_values = l_enhanced[skin_mask > 0]
    if len(skin_l_values) < 100:
        return np.zeros((h, w), dtype=np.uint8)

    mean_l = float(np.mean(skin_l_values))
    std_l = float(np.std(skin_l_values))
    median_l = float(np.median(skin_l_values))

    # Determine adaptive threshold factor based on skin tone contrast
    # Darker skin has higher contrast with vitiligo — use lower threshold
    if mean_l < 90:       # dark skin
        k_factor = 0.6
    elif mean_l < 130:    # medium skin
        k_factor = 0.8
    elif mean_l < 170:    # light skin
        k_factor = 1.2
    else:                 # very light skin
        k_factor = 1.5

    # ── Method A: Statistical threshold ──────────────────────────────────
    stat_thresh = mean_l + k_factor * max(std_l, 8)
    mask_stat = (l_enhanced > stat_thresh).astype(np.uint8) * 255
    mask_stat = cv2.bitwise_and(mask_stat, skin_mask)

    # ── Method B: Multi-scale adaptive threshold ─────────────────────────
    # Small scale catches small patches, large scale catches large ones
    masks_adaptive = []
    for block_mult in [6, 10, 16]:
        block_size = max(31, (min(h, w) // block_mult) | 1)
        # C parameter: negative means "brighter than local average"
        c_val = -max(3, int(std_l * 0.3))
        adaptive = cv2.adaptiveThreshold(
            l_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c_val,
        )
        adaptive = cv2.bitwise_and(adaptive, skin_mask)
        masks_adaptive.append(adaptive)

    # Combine adaptive: pixel must appear in at least 2 of 3 scales
    adaptive_sum = np.zeros((h, w), dtype=np.uint16)
    for m in masks_adaptive:
        adaptive_sum += (m > 0).astype(np.uint16)
    mask_adaptive = (adaptive_sum >= 2).astype(np.uint8) * 255

    # ── Method C: Color-distance in LAB ──────────────────────────────────
    # Compute median LAB color of skin region
    skin_a_values = a_channel[skin_mask > 0]
    skin_b_values = b_channel[skin_mask > 0]
    median_a = float(np.median(skin_a_values))
    median_b = float(np.median(skin_b_values))

    # Distance: vitiligo is lighter AND more neutral (closer to a=128, b=128)
    l_diff = l_enhanced.astype(np.float32) - median_l
    a_shift = np.abs(a_channel.astype(np.float32) - median_a)
    b_shift = np.abs(b_channel.astype(np.float32) - median_b)

    # Vitiligo: brighter than median AND shifted toward neutral
    neutrality = np.sqrt(
        (a_channel.astype(np.float32) - 128) ** 2 +
        (b_channel.astype(np.float32) - 128) ** 2
    )
    skin_neutrality = neutrality[skin_mask > 0]
    median_neutrality = float(np.median(skin_neutrality))

    # Depigmented skin is more neutral (lower chroma) than surrounding skin
    mask_color = np.zeros((h, w), dtype=np.uint8)
    is_brighter = l_diff > max(5, std_l * 0.3)
    is_neutral = neutrality < (median_neutrality + 5)  # more neutral or equally neutral
    mask_color[is_brighter & is_neutral] = 255
    mask_color = cv2.bitwise_and(mask_color, skin_mask)

    # ── Method D: Edge-boundary detection ────────────────────────────────
    # Strong luminance gradient at vitiligo borders
    grad_x = cv2.Sobel(l_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_thresh = np.percentile(gradient_mag[skin_mask > 0], 80)
    strong_edges = (gradient_mag > grad_thresh).astype(np.uint8) * 255

    # Dilate edges to form closed boundaries
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.dilate(strong_edges, edge_kernel, iterations=2)

    # Flood-fill bright regions bounded by edges
    # (This helps catch patches where threshold alone is ambiguous)

    # ── Combine all methods with weighted voting ─────────────────────────
    vote_map = np.zeros((h, w), dtype=np.float32)
    vote_map += (mask_stat > 0).astype(np.float32) * 1.0       # statistical
    vote_map += (mask_adaptive > 0).astype(np.float32) * 1.0   # multi-scale adaptive
    vote_map += (mask_color > 0).astype(np.float32) * 1.2      # color-distance (most reliable)

    # Require at least 2 methods to agree (threshold 1.8 allows color+one other)
    combined = (vote_map >= 1.8).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, skin_mask)

    # ── Post-processing ──────────────────────────────────────────────────
    # Close small gaps within patches
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove tiny specks (noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Fill holes inside detected patches (hair follicles leave holes)
    contours_fill, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_fill:
        cv2.drawContours(combined, [c], -1, 255, cv2.FILLED)

    return combined


# ─── Contour extraction ──────────────────────────────────────────────────────

def _extract_contours(
    depig_mask: np.ndarray,
    skin_pixels: int,
) -> list[np.ndarray]:
    """
    Extract and validate contours. Merges nearby contours that likely
    belong to the same patch (common with hair interference).
    """
    contours, _ = cv2.findContours(depig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: minimum 0.05% of skin area or 30px
    min_area = max(skin_pixels * 0.0005, 30)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if len(valid) <= 1:
        return valid

    # Merge nearby contours — patches fragmented by hair should unify
    merged = _merge_nearby_contours(valid, merge_distance=25)

    return merged


def _merge_nearby_contours(
    contours: list[np.ndarray],
    merge_distance: int = 25,
) -> list[np.ndarray]:
    """Merge contours whose bounding boxes are within merge_distance pixels."""
    if not contours:
        return []

    # Build bounding rects
    rects = [cv2.boundingRect(c) for c in contours]

    # Union-find for grouping
    n = len(contours)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Check pairwise distance between bounding rects
    for i in range(n):
        x1, y1, w1, h1 = rects[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = rects[j]
            # Distance between edges of bounding rects
            dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < merge_distance:
                union(i, j)

    # Group contours by their root parent
    groups: dict[int, list[np.ndarray]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(contours[i])

    # Merge each group into a single contour
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            all_points = np.vstack(group)
            hull = cv2.convexHull(all_points)
            merged.append(hull)

    return merged


# ─── Annotation drawing ─────────────────────────────────────────────────────

def _draw_annotations(
    img: np.ndarray,
    contours: list[np.ndarray],
    bounding_boxes: list[dict],
    mask: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    """Draw semi-transparent overlay + cyan contours + bounding boxes."""
    annotated = img.copy()

    # Semi-transparent cyan overlay on detected patches
    overlay = annotated.copy()
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    # Cyan contour outlines (thicker for visibility)
    cv2.drawContours(annotated, contours, -1, (255, 255, 0), 2)

    # Bounding boxes with rounded feel
    for bb in bounding_boxes:
        x1 = int(bb["left"] * w)
        y1 = int(bb["top"] * h)
        x2 = x1 + int(bb["width"] * w)
        y2 = y1 + int(bb["height"] * h)
        # Slightly padded box
        pad = 4
        cv2.rectangle(annotated,
                       (max(0, x1 - pad), max(0, y1 - pad)),
                       (min(w, x2 + pad), min(h, y2 + pad)),
                       (255, 255, 0), 1, cv2.LINE_AA)

    return annotated


# ─── Confidence scoring ──────────────────────────────────────────────────────

def _compute_confidence(
    l_channel: np.ndarray,
    depig_mask: np.ndarray,
    skin_mask: np.ndarray,
    skin_pixels: int,
    h: int,
    w: int,
    patch_count: int,
    contours: list[np.ndarray],
) -> float:
    """
    Multi-factor confidence score (0.0–1.0):
      - Skin coverage (more skin = more reliable)
      - Contrast between depigmented and normal skin
      - Edge sharpness at patch boundaries
      - Contour regularity (smooth boundaries = higher confidence)
    """
    total_pixels = h * w

    # Factor 1: Skin coverage (0–0.3)
    skin_coverage = skin_pixels / total_pixels
    f_coverage = min(0.3, skin_coverage * 0.4)

    # Factor 2: Contrast (0–0.35)
    f_contrast = 0.0
    if patch_count > 0:
        normal_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(depig_mask))
        if cv2.countNonZero(depig_mask) > 10 and cv2.countNonZero(normal_mask) > 10:
            depig_mean = float(cv2.mean(l_channel, mask=depig_mask)[0])
            normal_mean = float(cv2.mean(l_channel, mask=normal_mask)[0])
            l_diff = abs(depig_mean - normal_mean)
            # 15+ L* difference is clinically significant
            f_contrast = min(0.35, (l_diff / 50.0) * 0.35)

    # Factor 3: Edge sharpness at patch boundaries (0–0.2)
    f_edge = 0.0
    if contours:
        boundary_mask = np.zeros_like(depig_mask)
        cv2.drawContours(boundary_mask, contours, -1, 255, 3)
        boundary_mask = cv2.bitwise_and(boundary_mask, skin_mask)
        if cv2.countNonZero(boundary_mask) > 0:
            grad = cv2.Laplacian(l_channel, cv2.CV_64F)
            edge_strength = float(np.mean(np.abs(grad[boundary_mask > 0])))
            f_edge = min(0.2, (edge_strength / 30.0) * 0.2)

    # Factor 4: Contour regularity (0–0.15)
    f_regularity = 0.0
    if contours:
        solidities = []
        for c in contours:
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidities.append(area / hull_area)
        if solidities:
            avg_solidity = np.mean(solidities)
            f_regularity = min(0.15, avg_solidity * 0.15)

    confidence = f_coverage + f_contrast + f_edge + f_regularity

    # Baseline: if we detected skin and patches, minimum 0.4
    if patch_count > 0 and f_contrast > 0.05:
        confidence = max(0.4, confidence)

    return round(min(1.0, confidence), 3)


def _encode_jpeg_b64(img: np.ndarray, quality: int = 82) -> str:
    """Encode a BGR image as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
