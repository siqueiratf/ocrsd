from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    dpi: int = 300


    diff_threshold: int = 30
    morph_open_kernel: tuple[int, int] = (3, 3)
    min_component_area: int = 15
    min_component_height: int = 6


    word_dilate_kernel: tuple[int, int] = (15, 5)
    line_y_gap_ratio: float = 0.6


    ransac_reproj_threshold: float = 5.0
    min_match_count: int = 10
    min_inlier_ratio: float = 0.2


    ocr_engine: str = "easyocr"
    ocr_languages: list[str] = field(default_factory=lambda: ["en", "pt"])


    chunk_size: int = 50
    chunk_overlap: int = 2


    redact_color_bgr: tuple[int, int, int] = (0, 0, 255)
    redact_alpha: float = 1.0
    crop_padding: int = 4
