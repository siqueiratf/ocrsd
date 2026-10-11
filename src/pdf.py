import os
import tempfile
from dataclasses import dataclass

import fitz
import numpy as np


@dataclass
class PageImage:
    image: np.ndarray
    dpi: int
    width_px: int
    height_px: int
    page_size_pt: tuple[float, float]
    page_count: int


def keep_first_page_in_place(pdf_path: str) -> tuple[int, bool]:
    source = os.path.abspath(pdf_path)
    source_mode = os.stat(source).st_mode
    doc = fitz.open(source)
    temporary_path: str | None = None
    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError(f"'{pdf_path}' has no pages")
        if page_count == 1:
            return page_count, False
        doc.select([0])
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(source)}.", suffix=".tmp.pdf",
            dir=os.path.dirname(source),
        )
        os.close(descriptor)
        doc.save(temporary_path)
        os.chmod(temporary_path, source_mode)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    finally:
        doc.close()

    try:
        os.replace(temporary_path, source)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return page_count, True


def trim_to_first_page(pdf_path: str, output_path: str) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc)
        if page_count <= 1:
            return pdf_path, page_count
        doc.select([0])
        doc.save(output_path)
        return output_path, page_count
    finally:
        doc.close()


def pdf_to_image(pdf_path: str, dpi: int, page_number: int = 0) -> PageImage:
    doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            raise ValueError(f"'{pdf_path}' has no pages")
        page = doc[page_number]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )

        image_bgr = image[:, :, ::-1].copy()
        return PageImage(
            image=image_bgr,
            dpi=dpi,
            width_px=pixmap.width,
            height_px=pixmap.height,
            page_size_pt=(page.rect.width, page.rect.height),
            page_count=len(doc),
        )
    finally:
        doc.close()


def image_to_pdf(image: np.ndarray, dpi: int, output_pdf_path: str, jpeg_quality: int | None = None) -> None:
    height_px, width_px = image.shape[:2]
    width_pt = width_px * 72.0 / dpi
    height_pt = height_px * 72.0 / dpi

    ok, image_array = _encode_image(image, jpeg_quality)
    if not ok:
        raise RuntimeError("Failed to encode image for PDF")
    image_bytes = image_array.tobytes()

    doc = fitz.open()
    try:
        page = doc.new_page(width=width_pt, height=height_pt)
        page.insert_image(page.rect, stream=image_bytes)
        doc.save(output_pdf_path, deflate=True, garbage=3)
    finally:
        doc.close()


def _encode_image(image: np.ndarray, jpeg_quality: int | None):
    import cv2

    if jpeg_quality is not None:
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imencode(".png", image)


def standardize_size(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h = max(img_a.shape[0], img_b.shape[0])
    w = max(img_a.shape[1], img_b.shape[1])

    def pad(img: np.ndarray) -> np.ndarray:
        pad_h = h - img.shape[0]
        pad_w = w - img.shape[1]
        if pad_h == 0 and pad_w == 0:
            return img
        return cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    return pad(img_a), pad(img_b)
