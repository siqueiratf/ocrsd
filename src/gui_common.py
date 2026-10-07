from __future__ import annotations

import tkinter as tk


WHEEL_SEQUENCES = (
    "<MouseWheel>",
    "<Shift-MouseWheel>",
    "<Button-4>",
    "<Button-5>",
    "<Shift-Button-4>",
    "<Shift-Button-5>",
)
VALUE_WIDGET_CLASSES = ("TCombobox", "TSpinbox", "TScale", "Spinbox", "Scale")


def _ignore_wheel(_event: tk.Event) -> None:
    return None


def disable_wheel_value_changes(widget: tk.Misc) -> None:
    for widget_class in VALUE_WIDGET_CLASSES:
        for sequence in WHEEL_SEQUENCES:
            widget.bind_class(widget_class, sequence, _ignore_wheel)
