import cv2
import numpy as np

INK_HUE_RANGE = (80, 140)
INK_SAT_MIN = 10
INK_GRAY_MAX = 220
MIN_INK_COMPONENT_AREA = 4


def blue_ink_mask(image: np.ndarray, hue_range: tuple[int, int] = INK_HUE_RANGE,
                  sat_min: int = INK_SAT_MIN, gray_max: int = INK_GRAY_MAX,
                  min_component_area: int = MIN_INK_COMPONENT_AREA,
                  close_kernel: int = 3) -> np.ndarray:
    if image.ndim == 2:
        return np.where(image < gray_max, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hue, sat, _ = cv2.split(hsv)
    mask = (
        (gray < gray_max)
        & (hue >= hue_range[0])
        & (hue <= hue_range[1])
        & (sat >= sat_min)
    ).astype(np.uint8) * 255


    if close_kernel > 1:
        close_kernel = close_kernel if close_kernel % 2 else close_kernel + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if min_component_area > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned[labels == label] = 255
        mask = cleaned
    return mask


def remove_form_lines(image: np.ndarray,
                      hue_range: tuple[int, int] = INK_HUE_RANGE,
                      sat_min: int = INK_SAT_MIN,
                      gray_max: int = INK_GRAY_MAX) -> np.ndarray:
    mask = blue_ink_mask(
        image, hue_range=hue_range, sat_min=sat_min, gray_max=gray_max,
        min_component_area=1, close_kernel=1,
    )
    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result


def convert_ink_to_black(image: np.ndarray, ink_threshold: int = 250) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    result = gray.copy()
    result[gray < ink_threshold] = 0
    return result


def remove_ink_noise(image: np.ndarray, min_component_area: int = 4,
                     close_kernel: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = np.where(gray < 128, 255, 0).astype(np.uint8)
    if close_kernel > 1:
        close_kernel = close_kernel if close_kernel % 2 else close_kernel + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= max(1, min_component_area):
            cleaned[labels == label] = 255
    return np.where(cleaned > 0, 0, 255).astype(np.uint8)


def whiten_background(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.where(gray < threshold, 0, 255).astype(np.uint8)


def adjust_tone(image: np.ndarray, brightness: float = 0,
                contrast: float = 0, exposure: float = 0) -> np.ndarray:
    contrast_factor = max(0.0, 1.0 + contrast / 100.0)
    exposure_factor = 2.0 ** exposure
    adjusted = image.astype(np.float32) * contrast_factor * exposure_factor + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def isolate_ink_hsv(image: np.ndarray, hue_range: tuple[int, int] = INK_HUE_RANGE,
                     sat_min: int = INK_SAT_MIN, gray_max: int = INK_GRAY_MAX) -> np.ndarray:
    return remove_form_lines(image, hue_range, sat_min, gray_max)


def denoise(image: np.ndarray, method: str = "nlmeans") -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(image, 3)
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                                templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0,
                            tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def binarize(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 25, 10)


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(ink)
    if coords is None or len(coords) < 20:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border_value = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def pad(image: np.ndarray, margin: int = 15) -> np.ndarray:
    border_value = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.copyMakeBorder(image, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=border_value)


def preprocess_for_ocr(image: np.ndarray, mode: str = "traditional",
                        isolate_ink: bool = True, margin: int = 15,
                        hue_range: tuple[int, int] = INK_HUE_RANGE,
                        sat_min: int = INK_SAT_MIN,
                        gray_max: int = INK_GRAY_MAX,
                        min_component_area: int = MIN_INK_COMPONENT_AREA,
                        close_kernel: int = 3,
                        ink_threshold: int = 250,
                        background_threshold: int = 245,
                        brightness: float = 0,
                        contrast: float = 0,
                        exposure: float = 0,
                        grayscale_only: bool = False) -> np.ndarray:
    if mode not in {"traditional", "vlm"}:
        raise ValueError(f"Modo de pré-processamento desconhecido: {mode}")

    image = adjust_tone(image, brightness, contrast, exposure)
    if grayscale_only:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


    if mode == "traditional" and isolate_ink and image.ndim == 3:
        out = remove_form_lines(image, hue_range, sat_min, gray_max)
        out = convert_ink_to_black(out, ink_threshold)
        out = remove_ink_noise(out, min_component_area, close_kernel)
        out = whiten_background(out, background_threshold)
        out = deskew(out)
        return pad(out, margin=margin)

    out = isolate_ink_hsv(image) if isolate_ink else image
    out = denoise(out, method="nlmeans")
    out = enhance_contrast_clahe(out)
    out = deskew(out)
    if mode == "traditional":
        out = binarize(out, method="adaptive")
    out = pad(out, margin=margin)
    return out
