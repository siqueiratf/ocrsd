from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
FONTS_DIR = ASSETS_ROOT / "fonts"


@dataclass(frozen=True)
class FontFamily:
    key: str
    label: str
    category: str  # "handwriting" ou "typed"
    filename: str


FONT_FAMILIES: dict[str, FontFamily] = {
    "caveat": FontFamily("caveat", "Caveat", "handwriting", "Caveat-Regular.ttf"),
    "homemade_apple": FontFamily("homemade_apple", "Homemade Apple", "handwriting", "HomemadeApple-Regular.ttf"),
    "courier_prime": FontFamily("courier_prime", "Courier Prime", "typed", "CourierPrime-Regular.ttf"),
    "arimo": FontFamily("arimo", "Arimo", "typed", "Arimo-Regular.ttf"),
    "tinos": FontFamily("tinos", "Tinos", "typed", "Tinos-Regular.ttf"),
}
DEFAULT_FONT_FAMILY = "caveat"
FONT_FAMILY_KEYS = tuple(FONT_FAMILIES)
HANDWRITING_FAMILIES = tuple(key for key, item in FONT_FAMILIES.items() if item.category == "handwriting")
TYPED_FAMILIES = tuple(key for key, item in FONT_FAMILIES.items() if item.category == "typed")


def font_family(key: str | None) -> FontFamily:
    return FONT_FAMILIES.get(key or DEFAULT_FONT_FAMILY, FONT_FAMILIES[DEFAULT_FONT_FAMILY])


def font_path(key: str | None) -> Path:
    return FONTS_DIR / font_family(key).filename


def is_typed(key: str | None) -> bool:
    return font_family(key).category == "typed"
