import argparse
import json
import os
import sys

import cv2

import numpy as np

from src.config import PipelineConfig
from src import alignment, difference, fields as fields_module, ocr, pdf, redaction, segmentation
from src.alignment import AlignmentResult
from src.fields import FieldMap
from src.segmentation import WordRegion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and redact filled-in content in a civil registry document."
    )
    parser.add_argument("--document", required=True, help="Path to the completed (filled-in) PDF")
    parser.add_argument("--template", required=True, help="Path to the blank template PDF")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--document-id", default="document_001", help="Identifier used in output filenames")
    parser.add_argument(
        "--field-map", default=None,
        help="Path to a field-map JSON (see documents/<tipo>/mappings/). When given, "
             "detection is restricted to the declared field boxes and output is per-field "
             "instead of per-word."
    )
    parser.add_argument(
        "--skip-ocr", action="store_true",
        help="Skip OCR (ocr_text left empty, confidence 0.0). Much faster; useful for "
             "iterating on alignment/detection alone."
    )
    parser.add_argument(
        "--stop-after", choices=["raster", "align", "diff"], default=None,
        help="Stop after this stage and only write its debug artifacts (no OCR/redaction/"
             "manifest). Useful for quickly checking one stage in isolation."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI; must match the field map")
    parser.add_argument("--ocr-engine", default="easyocr", help="OCR backend name (see ocrsd models)")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _prepare_output_dirs(output_dir: str) -> dict[str, str]:
    dirs = {
        "root": output_dir,
        "masked": os.path.join(output_dir, "masked"),
        "crops": os.path.join(output_dir, "crops"),
        "debug": os.path.join(output_dir, "debug"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def run_pipeline(document_pdf: str, template_pdf: str, output_dir: str,
                  config: PipelineConfig, document_id: str = "document_001",
                  field_map_path: str | None = None, skip_ocr: bool = False,
                  stop_after: str | None = None) -> None:
    dirs = _prepare_output_dirs(output_dir)

    if not os.path.isfile(document_pdf):
        raise FileNotFoundError(f"Completed document not found: {document_pdf}")
    if not os.path.isfile(template_pdf):
        raise FileNotFoundError(f"Template not found: {template_pdf}")
    if field_map_path is not None and not os.path.isfile(field_map_path):
        raise FileNotFoundError(f"Field map not found: {field_map_path}")

    field_map: FieldMap | None = (
        fields_module.load_field_map(field_map_path) if field_map_path else None
    )
    if config.dpi <= 0:
        raise ValueError("DPI must be greater than zero")
    if field_map is not None and field_map.dpi != config.dpi:
        raise ValueError(f"Field map uses {field_map.dpi} DPI; use --dpi {field_map.dpi}")

    original_document_pdf, original_template_pdf = document_pdf, template_pdf
    document_pdf, document_page_count = pdf.trim_to_first_page(
        document_pdf, os.path.join(dirs["debug"], "document_page1.pdf")
    )
    template_pdf, template_page_count = pdf.trim_to_first_page(
        template_pdf, os.path.join(dirs["debug"], "template_page1.pdf")
    )
    if document_page_count > 1:
        print(f"'{original_document_pdf}' had {document_page_count} pages; "
              f"kept only page 1 (saved to '{document_pdf}'), dropped the rest.", file=sys.stderr)
    if template_page_count > 1:
        print(f"'{original_template_pdf}' had {template_page_count} pages; "
              f"kept only page 1 (saved to '{template_pdf}'), dropped the rest.", file=sys.stderr)

    completed_page = pdf.pdf_to_image(document_pdf, dpi=config.dpi, page_number=0)
    template_page = pdf.pdf_to_image(template_pdf, dpi=config.dpi, page_number=0)

    completed_img, template_img = pdf.standardize_size(completed_page.image, template_page.image)

    cv2.imwrite(os.path.join(dirs["debug"], "completed_raw.png"), completed_img)
    cv2.imwrite(os.path.join(dirs["debug"], "template_raw.png"), template_img)

    print(f"Rasterized '{document_pdf}' and '{template_pdf}' at {config.dpi} DPI "
          f"({completed_img.shape[1]}x{completed_img.shape[0]}px).")

    if stop_after == "raster":
        print(f"Stopped after 'raster'. Debug images written to {dirs['debug']}")
        return

    try:
        align_result = alignment.align_to_template(template_img, completed_img, config)
    finally:
        alignment.save_match_debug_image(
            template_img, completed_img, out_path=os.path.join(dirs["debug"], "matches.png")
        )

    cv2.imwrite(os.path.join(dirs["debug"], "aligned.png"), align_result.warped_image)
    print(f"Alignment: {align_result.match_count} matches, "
          f"inlier ratio {align_result.inlier_ratio:.2f}, scale {align_result.scale:.2f}x")

    if stop_after == "align":
        print(f"Stopped after 'align'. Debug images written to {dirs['debug']}")
        return

    search_mask = align_result.valid_mask
    if field_map is not None:
        field_mask = fields_module.build_field_mask(field_map, completed_img.shape)
        search_mask = cv2.bitwise_and(search_mask, field_mask)
        cv2.imwrite(os.path.join(dirs["debug"], "field_mask.png"), field_mask)
        cv2.imwrite(
            os.path.join(dirs["debug"], "field_overlay.png"),
            draw_field_overlay(align_result.warped_image, field_map),
        )

    open_kernel_size = difference.adaptive_open_kernel(align_result.scale, config.morph_open_kernel)
    if open_kernel_size != tuple(config.morph_open_kernel):
        print(f"Scale {align_result.scale:.2f}x detected; using a larger denoise kernel "
              f"{open_kernel_size} (base {tuple(config.morph_open_kernel)}) to absorb "
              f"resampling blur from the rescale.")
    diff_result = difference.run_difference(
        template_img, align_result.warped_image, search_mask, config, open_kernel_size
    )
    cv2.imwrite(os.path.join(dirs["debug"], "diff.png"), diff_result.diff_gray)
    cv2.imwrite(os.path.join(dirs["debug"], "mask.png"), diff_result.binary_mask)

    if stop_after == "diff":
        print(f"Stopped after 'diff'. Debug images written to {dirs['debug']}")
        return

    if field_map is not None:
        masked_image, manifest = run_field_pipeline(
            document_id, document_pdf, template_pdf, align_result, diff_result,
            field_map, dirs, config, skip_ocr
        )
    else:
        masked_image, manifest = run_word_pipeline(
            document_id, document_pdf, template_pdf, align_result, diff_result,
            completed_img.shape, dirs, config, skip_ocr
        )

    masked_path = os.path.join(dirs["masked"], f"{document_id}.png")
    cv2.imwrite(masked_path, masked_image)

    output_pdf_path = os.path.join(dirs["root"], f"{document_id}.pdf")
    redaction.build_output_pdf(masked_image, config.dpi, output_pdf_path)

    manifest_path = os.path.join(dirs["root"], f"{document_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Masked image: {masked_path}")
    print(f"Redacted PDF: {output_pdf_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Debug images written to {dirs['debug']}")


def run_word_pipeline(document_id, document_pdf, template_pdf, align_result, diff_result,
                       image_shape, dirs, config: PipelineConfig, skip_ocr: bool = False):
    words = segmentation.group_into_words(
        diff_result.binary_mask, config.word_dilate_kernel, config.min_component_area
    )
    words = segmentation.group_into_lines(words, config.line_y_gap_ratio)
    words = segmentation.reading_order_sort(words)
    print(f"Detected {len(words)} filled-content regions.")

    if skip_ocr:
        word_ocr = {w.word_id: ocr.OcrResult(text="", confidence=0.0) for w in words}
        line_ocr = {w.line_id: ocr.OcrResult(text="", confidence=0.0) for w in words}
    else:
        word_ocr, line_ocr = run_ocr(align_result.warped_image, words, config)
        print(f"OCR complete for {len(words)} words across {len(line_ocr)} lines.")

    masked_image = redaction.draw_redactions(
        align_result.warped_image, [w.bbox for w in words], config.redact_color_bgr, config.redact_alpha
    )
    crop_paths = redaction.save_crops(
        align_result.warped_image, words, dirs["crops"], config.crop_padding
    )

    manifest = build_manifest(
        document_id=document_id,
        document_pdf=document_pdf,
        template_pdf=template_pdf,
        words=words,
        crop_paths=crop_paths,
        word_ocr=word_ocr,
        line_ocr=line_ocr,
        alignment_result=align_result,
        image_shape=image_shape,
        config=config,
    )
    return masked_image, manifest


def run_field_pipeline(document_id, document_pdf, template_pdf, align_result, diff_result,
                        field_map: FieldMap, dirs, config: PipelineConfig, skip_ocr: bool = False):
    components = segmentation.find_components(diff_result.binary_mask, config.min_component_area)
    assigned = fields_module.assign_components_to_fields(components, field_map)

    ordered_fields = sorted(field_map.fields, key=lambda f: (f.bbox[1], f.bbox[0]))
    field_ids = [f.field_id for f in ordered_fields]
    field_boxes = {f.field_id: f.bbox for f in ordered_fields}
    field_labels = {f.field_id: f.label for f in ordered_fields}

    field_ocr: dict[str, "ocr.OcrResult"] = {}
    element_results: dict[str, list[dict]] = {}
    filled_ids: list[str] = []
    for f in ordered_fields:
        is_filled = f.always_redact or len(assigned.get(f.field_id, [])) > 0
        child_assignments = fields_module.assign_components_to_elements(
            assigned.get(f.field_id, []), f,
        ) if f.elements else {}
        child_results: list[dict] = []
        if f.elements:
            ordered_elements = sorted(
                f.elements, key=lambda element: (
                    element.index if element.index is not None else 10_000,
                    element.bbox[1], element.bbox[0],
                ),
            )
            for element in ordered_elements:
                element_filled = bool(child_assignments.get(element.element_id))
                text, confidence = "", 0.0
                if element_filled and not skip_ocr and f.data_type in {"digit_sequence", "text_sequence"}:
                    element_crop = ocr.crop_region(
                        align_result.warped_image, element.bbox, config.crop_padding,
                    )
                    prediction = ocr.recognize_word(element_crop, config)
                    text, confidence = prediction.text, prediction.confidence
                child_results.append({
                    "element_id": element.element_id,
                    "label": element.label,
                    "value": element.value,
                    "index": element.index,
                    "bbox": list(element.bbox),
                    "filled": element_filled,
                    "ocr_text": text,
                    "confidence": confidence,
                })
            element_results[f.field_id] = child_results
        if is_filled:
            filled_ids.append(f.field_id)
            if skip_ocr:
                field_ocr[f.field_id] = ocr.OcrResult(text="", confidence=0.0)
            elif f.data_type == "checkbox":
                selected = [
                    str(item.get("value") or item["element_id"])
                    for item in child_results if item["filled"]
                ]
                field_ocr[f.field_id] = ocr.OcrResult(text=",".join(selected), confidence=1.0 if selected else 0.0)
            elif f.data_type in {"digit_sequence", "text_sequence"}:
                texts = [item["ocr_text"] for item in child_results]
                confidences = [item["confidence"] for item in child_results if item["ocr_text"]]
                field_ocr[f.field_id] = ocr.OcrResult(
                    text="".join(texts),
                    confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                )
            else:
                crop = ocr.crop_region(align_result.warped_image, f.bbox, config.crop_padding)
                field_ocr[f.field_id] = ocr.recognize_line(crop, config)
        else:
            field_ocr[f.field_id] = ocr.OcrResult(text="", confidence=0.0)
    print(f"Detected content in {len(filled_ids)}/{len(ordered_fields)} mapped fields.")

    masked_image = redaction.draw_redactions(
        align_result.warped_image,
        [field_boxes[fid] for fid in filled_ids],
        config.redact_color_bgr, config.redact_alpha,
    )
    crop_paths = redaction.save_field_crops(
        align_result.warped_image, field_ids, field_labels, field_boxes,
        dirs["crops"], config.crop_padding,
    )

    manifest = build_field_manifest(
        document_id=document_id,
        document_pdf=document_pdf,
        template_pdf=template_pdf,
        ordered_fields=ordered_fields,
        filled_ids=set(filled_ids),
        field_ocr=field_ocr,
        element_results=element_results,
        crop_paths=crop_paths,
        alignment_result=align_result,
        field_map=field_map,
        config=config,
    )
    return masked_image, manifest


def draw_field_overlay(image: np.ndarray, field_map: FieldMap) -> np.ndarray:
    overlay = image.copy()
    for f in field_map.fields:
        x1, y1, x2, y2 = f.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 120, 0), thickness=2)
        cv2.putText(overlay, f.field_id, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 120, 0), 2, cv2.LINE_AA)
        for element in f.elements or []:
            ex1, ey1, ex2, ey2 = element.bbox
            cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (30, 180, 255), thickness=2)
    return overlay


def run_ocr(image, words: list[WordRegion], config: PipelineConfig):
    word_ocr: dict[int, "ocr.OcrResult"] = {}
    for word in words:
        crop = ocr.crop_region(image, word.bbox, config.crop_padding)
        word_ocr[word.word_id] = ocr.recognize_word(crop, config)

    lines: dict[int, list[WordRegion]] = {}
    for word in words:
        lines.setdefault(word.line_id, []).append(word)

    line_ocr: dict[int, "ocr.OcrResult"] = {}
    for line_id, line_words in lines.items():
        strip = ocr.build_line_strip(image, line_words, config.crop_padding)
        line_ocr[line_id] = ocr.recognize_line(strip, config)

    return word_ocr, line_ocr


def chunk_entries(entries: list[dict], chunk_size: int, overlap: int, id_field: str) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")

    n = len(entries)
    if n == 0:
        return []

    total_chunks = -(-n // chunk_size)
    chunks = []
    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_size
        end = min(n, start + chunk_size)
        for entry in entries[start:end]:
            entry["chunk_id"] = chunk_id

        overlap_start = max(0, start - overlap)
        overlap_end = min(n, end + overlap)
        chunks.append({
            "chunk_id": chunk_id,
            f"{id_field}s": [e[id_field] for e in entries[overlap_start:overlap_end]],
        })

    return chunks


def build_manifest(document_id: str, document_pdf: str, template_pdf: str,
                    words: list[WordRegion], crop_paths: dict[int, str],
                    word_ocr: dict, line_ocr: dict,
                    alignment_result: AlignmentResult, image_shape: tuple,
                    config: PipelineConfig) -> dict:
    height, width = image_shape[:2]
    word_entries = []
    for word in words:
        w_ocr = word_ocr.get(word.word_id)
        l_ocr = line_ocr.get(word.line_id)
        word_entries.append({
            "document_id": document_id,
            "word_id": word.word_id,
            "page": 1,
            "line_id": word.line_id,
            "bbox": list(word.bbox),
            "classification": "filled_content",
            "ocr_text": w_ocr.text if w_ocr else "",
            "confidence": w_ocr.confidence if w_ocr else 0.0,
            "line_ocr_text": l_ocr.text if l_ocr else "",
            "crop_path": crop_paths.get(word.word_id, ""),
            "chunk_id": 0,
        })

    chunks = chunk_entries(word_entries, config.chunk_size, config.chunk_overlap, "word_id")

    return {
        "document_id": document_id,
        "source_files": {"completed": document_pdf, "template": template_pdf},
        "page": 1,
        "page_size": {"width_px": width, "height_px": height, "dpi": config.dpi},
        "alignment": {
            "success": alignment_result.success,
            "match_count": alignment_result.match_count,
            "inlier_ratio": alignment_result.inlier_ratio,
            "scale": alignment_result.scale,
            "homography": alignment_result.homography.tolist(),
        },
        "config_snapshot": {
            "diff_threshold": config.diff_threshold,
            "min_component_area": config.min_component_area,
            "min_component_height": config.min_component_height,
            "ocr_engine": config.ocr_engine,
        },
        "words": word_entries,
        "chunking": {
            "chunk_size": config.chunk_size,
            "overlap": config.chunk_overlap,
            "total_chunks": len(chunks),
            "chunks": chunks,
        },
    }


def build_field_manifest(document_id: str, document_pdf: str, template_pdf: str,
                          ordered_fields: list, filled_ids: set, field_ocr: dict,
                          element_results: dict[str, list[dict]],
                          crop_paths: dict[str, str], alignment_result: AlignmentResult,
                          field_map: FieldMap, config: PipelineConfig) -> dict:
    field_entries = []
    for f in ordered_fields:
        f_ocr = field_ocr.get(f.field_id)
        field_entries.append({
            "document_id": document_id,
            "field_id": f.field_id,
            "label": f.label,
            "description": f.description,
            "page": 1,
            "bbox": list(f.bbox),
            "filled": f.field_id in filled_ids,
            "classification": "filled_content" if f.field_id in filled_ids else "empty",
            "detection_method": "always_redact" if f.always_redact else "diff",
            "ocr_text": f_ocr.text if f_ocr else "",
            "confidence": f_ocr.confidence if f_ocr else 0.0,
            "data_type": f.data_type,
            "validation": f.validation or {},
            "selection_mode": f.selection_mode,
            "elements": element_results.get(f.field_id, []),
            "crop_path": crop_paths.get(f.field_id, ""),
            "chunk_id": 0,
        })

    chunks = chunk_entries(field_entries, config.chunk_size, config.chunk_overlap, "field_id")

    return {
        "document_id": document_id,
        "template_id": field_map.template_id,
        "source_files": {"completed": document_pdf, "template": template_pdf},
        "page": 1,
        "page_size": {
            "width_px": field_map.page_size_px[0],
            "height_px": field_map.page_size_px[1],
            "dpi": config.dpi,
        },
        "alignment": {
            "success": alignment_result.success,
            "match_count": alignment_result.match_count,
            "inlier_ratio": alignment_result.inlier_ratio,
            "scale": alignment_result.scale,
            "homography": alignment_result.homography.tolist(),
        },
        "config_snapshot": {
            "diff_threshold": config.diff_threshold,
            "min_component_area": config.min_component_area,
            "min_component_height": config.min_component_height,
            "ocr_engine": config.ocr_engine,
        },
        "fields": field_entries,
        "chunking": {
            "chunk_size": config.chunk_size,
            "overlap": config.chunk_overlap,
            "total_chunks": len(chunks),
            "chunks": chunks,
        },
    }


def run_from_args(args: argparse.Namespace) -> None:
    config = PipelineConfig(dpi=args.dpi, ocr_engine=args.ocr_engine)
    try:
        run_pipeline(args.document, args.template, args.output, config, args.document_id,
                     field_map_path=args.field_map, skip_ocr=args.skip_ocr,
                     stop_after=args.stop_after)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main(argv=None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
