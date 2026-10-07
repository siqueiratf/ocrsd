from dataclasses import dataclass

import cv2
import numpy as np

from src.config import PipelineConfig


class AlignmentError(Exception):
    pass


@dataclass
class AlignmentResult:
    warped_image: np.ndarray
    homography: np.ndarray
    valid_mask: np.ndarray
    match_count: int
    inlier_ratio: float
    scale: float
    success: bool


def estimate_scale(homography: np.ndarray) -> float:
    singular_values = np.linalg.svd(homography[:2, :2], compute_uv=False)
    return float(np.sqrt(singular_values[0] * singular_values[1]))


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _match_features(template_gray: np.ndarray, completed_gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=5000)
    kp_template, des_template = orb.detectAndCompute(template_gray, None)
    kp_completed, des_completed = orb.detectAndCompute(completed_gray, None)

    if des_template is None or des_completed is None:
        return kp_template, kp_completed, []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_completed, des_template, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp_template, kp_completed, good_matches


def align_to_template(template_img: np.ndarray, completed_img: np.ndarray,
                       config: PipelineConfig) -> AlignmentResult:
    template_gray = _to_gray(template_img)
    completed_gray = _to_gray(completed_img)

    kp_template, kp_completed, good_matches = _match_features(template_gray, completed_gray)

    if len(good_matches) < config.min_match_count:
        raise AlignmentError(
            f"Only {len(good_matches)} good feature matches found "
            f"(need >= {config.min_match_count}). The completed document may not "
            f"correspond to the given template, or the scan quality is too low."
        )

    src_pts = np.float32([kp_completed[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, config.ransac_reproj_threshold
    )

    if homography is None:
        raise AlignmentError("Could not estimate a homography from the matched features.")

    inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)
    if inlier_ratio < config.min_inlier_ratio:
        raise AlignmentError(
            f"Homography inlier ratio {inlier_ratio:.2f} is below the required "
            f"{config.min_inlier_ratio:.2f}. Alignment is unreliable."
        )

    h, w = template_gray.shape[:2]
    warped_image = cv2.warpPerspective(
        completed_img, homography, (w, h), borderValue=(255, 255, 255)
    )

    ones = np.full(completed_gray.shape, 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(ones, homography, (w, h), borderValue=0)

    return AlignmentResult(
        warped_image=warped_image,
        homography=homography,
        valid_mask=valid_mask,
        match_count=len(good_matches),
        inlier_ratio=inlier_ratio,
        scale=estimate_scale(homography),
        success=True,
    )


def save_match_debug_image(template_img: np.ndarray, completed_img: np.ndarray,
                            template_gray: np.ndarray = None, completed_gray: np.ndarray = None,
                            out_path: str = None) -> None:
    t_gray = _to_gray(template_img)
    c_gray = _to_gray(completed_img)
    kp_template, kp_completed, good_matches = _match_features(t_gray, c_gray)

    debug_img = cv2.drawMatches(
        completed_img, kp_completed, template_img, kp_template,
        good_matches[:200], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(out_path, debug_img)
