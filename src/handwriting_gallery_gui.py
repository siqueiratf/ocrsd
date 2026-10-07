from __future__ import annotations

import io
import queue
import random
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import colorchooser, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from src.neural_handwriting import NeuralPen, load_model, prepare_text, render_strokes
from src.synthetic_dataset import (
    render_handwritten_numeric, render_handwritten_text, random_handwriting_style,
)
from src.synthetic_dataset_gui import MODEL_PROFILES, MODEL_WRITER_MAX
from src.writing_presets import generation_settings, new_preset_uid, suggested_preset_name, upsert_preset


PANEL = "#202328"
PANEL_2 = "#292d33"
TEXT = "#f2f3f5"
MUTED = "#a9afb8"
PEN_WIDTH_MAX = 40.0

SAMPLE_TEXTS = (
    ("Data", "15092026"),
    ("UF", "SP"),
    ("Nome", "Ana Silva Santos"),
    ("CEP", "01001000"),
    ("Número", "042"),
)
VARIATION_DEFAULTS = (
    {"color": "#193787", "width": 1.4, "pressure": 0.20, "dropout": 0.00},
    {"color": "#1f1f24", "width": 2.1, "pressure": 0.45, "dropout": 0.04},
    {"color": "#315f9d", "width": 1.0, "pressure": 0.65, "dropout": 0.10},
)


@dataclass(frozen=True)
class GalleryVariation:
    color: str
    width: float
    pressure: float
    dropout: float


class HandwritingGalleryFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.model_var = tk.StringVar(value="font")
        self.writer_var = tk.StringVar(value="Aleatório")
        self.bias_var = tk.DoubleVar(value=0.75)
        self.text_policy_var = tk.StringVar(value="normalize")
        self.status_var = tk.StringVar(value="Escolha um modelo e gere a galeria.")
        self.variation_vars: list[dict[str, tk.Variable]] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.gallery_images: list[ImageTk.PhotoImage] = []
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Panel.TFrame", padding=10)
        header.pack(fill="x", pady=(10, 8))
        ttk.Label(header, text="Galeria de escrita", style="Panel.TLabel", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var, style="Panel.TLabel", anchor="center").pack(side="left", fill="x", expand=True, padx=12)
        self.generate_button = ttk.Button(header, text="Gerar galeria", style="Accent.TButton", command=self.generate_gallery)
        self.generate_button.pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, style="Panel.TFrame", width=340)
        preview = ttk.Frame(body)
        body.add(controls, weight=1)
        body.add(preview, weight=4)

        ttk.Label(controls, text="Modelo", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.model_box = ttk.Combobox(
            controls, textvariable=self.model_var,
            values=("font", "sjvasquez", "aaron1259", "wpvnteam"),
            state="readonly",
        )
        self.model_box.pack(fill="x", padx=12, pady=(0, 8))
        self.model_box.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(controls, text="Estilo", style="Panel.TLabel").pack(anchor="w", padx=12)
        self.writer_box = ttk.Combobox(controls, textvariable=self.writer_var, state="readonly")
        self.writer_box.pack(fill="x", padx=12, pady=(2, 8))
        self._spin_row(controls, "Regularidade", self.bias_var, 0, 3, 0.05)
        ttk.Label(controls, text="Caracteres não suportados", style="Panel.TLabel").pack(anchor="w", padx=12, pady=(6, 0))
        ttk.Combobox(controls, textvariable=self.text_policy_var, values=("strict", "normalize"), state="readonly").pack(fill="x", padx=12, pady=(2, 14))

        ttk.Label(controls, text="Variações da caneta", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        for index, defaults in enumerate(VARIATION_DEFAULTS, start=1):
            frame = ttk.Frame(controls, style="Panel.TFrame", padding=(12, 8))
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"Variação {index}", style="Panel.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
            variables: dict[str, tk.Variable] = {
                "color": tk.StringVar(value=defaults["color"]),
                "width": tk.DoubleVar(value=defaults["width"]),
                "pressure": tk.DoubleVar(value=defaults["pressure"]),
                "dropout": tk.DoubleVar(value=defaults["dropout"]),
            }
            swatch = tk.Button(frame, bg=str(defaults["color"]), width=4, command=lambda vars=variables: self._choose_color(vars))
            swatch.grid(row=1, column=1, sticky="e", pady=2)
            ttk.Label(frame, text="Cor", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
            variables["swatch"] = swatch
            ttk.Label(frame, text="Espessura", style="Panel.TLabel").grid(row=2, column=0, sticky="w")
            width_value = ttk.Label(frame, text=f"{float(variables['width'].get()):.2f}", style="Panel.TLabel", width=5, anchor="e")
            width_value.grid(row=2, column=1, sticky="e")
            width_scale = ttk.Scale(frame, from_=0.25, to=PEN_WIDTH_MAX, variable=variables["width"])
            width_scale.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))
            variables["width"].trace_add(
                "write",
                lambda *_args, shown=width_value, var=variables["width"]: shown.configure(text=f"{float(var.get()):.2f}"),
            )
            for row, (key, label, low, high) in enumerate((
                ("pressure", "Pressão", 0, 1),
                ("dropout", "Falhas", 0, 1),
            ), start=4):
                ttk.Label(frame, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w")
                ttk.Spinbox(frame, from_=low, to=high, increment=0.05, textvariable=variables[key], width=8).grid(row=row, column=1, sticky="e", pady=2)
            ttk.Button(
                frame, text="Salvar preset",
                command=lambda vars=variables, idx=index: self._save_preset(idx, vars),
            ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            frame.columnconfigure(0, weight=1)
            self.variation_vars.append(variables)

        self.canvas = tk.Canvas(preview, bg="#0e1012", highlightthickness=0)
        scrollbar = ttk.Scrollbar(preview, orient="vertical", command=self.canvas.yview)
        self.gallery_frame = ttk.Frame(self.canvas)
        self.gallery_window = self.canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.gallery_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.gallery_window, width=event.width))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._model_changed()

    def _spin_row(self, parent: tk.Misc, label: str, variable: tk.DoubleVar,
                  low: float, high: float, increment: float) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", padx=12, pady=3)
        ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
        ttk.Spinbox(row, from_=low, to=high, increment=increment, textvariable=variable, width=8).pack(side="right")

    def _model_changed(self, _event=None) -> None:
        model = self.model_var.get()
        profile = MODEL_PROFILES[model]
        self.bias_var.set(float(profile["bias"]))
        self.text_policy_var.set(str(profile["text_policy"]))
        if model == "font":
            self.writer_box.configure(values=("Aleatório",), state="disabled")
            self.writer_var.set("Aleatório")
        else:
            maximum = MODEL_WRITER_MAX[model]
            self.writer_box.configure(values=("Aleatório", *(str(index) for index in range(maximum + 1))), state="readonly")
            if self.writer_var.get() != "Aleatório":
                try:
                    writer = int(self.writer_var.get())
                except ValueError:
                    writer = -1
                if not 0 <= writer <= maximum:
                    self.writer_var.set("Aleatório")

    def _choose_color(self, variables: dict[str, tk.Variable]) -> None:
        _rgb, color = colorchooser.askcolor(color=variables["color"].get(), parent=self)
        if color:
            variables["color"].set(color)
            variables["swatch"].configure(bg=color)

    def _variations(self) -> list[GalleryVariation]:
        result = []
        for variables in self.variation_vars:
            result.append(GalleryVariation(
                color=str(variables["color"].get()),
                width=max(0.25, min(PEN_WIDTH_MAX, float(variables["width"].get()))),
                pressure=max(0.0, min(1.0, float(variables["pressure"].get()))),
                dropout=max(0.0, min(1.0, float(variables["dropout"].get()))),
            ))
        return result

    def _save_preset(self, index: int, variables: dict[str, tk.Variable]) -> None:
        settings = generation_settings(
            model=self.model_var.get(),
            writer=self.writer_var.get(),
            bias=float(self.bias_var.get()),
            pen_color=str(variables["color"].get()),
            pen_width=float(variables["width"].get()),
            pen_pressure=float(variables["pressure"].get()),
            pen_dropout=float(variables["dropout"].get()),
            text_policy=self.text_policy_var.get(),
            handwriting=True,
            text_handwriting=True,
            handwriting_intensity=0.65,
            realistic_filters=False,
            filter_preset="medium",
        )
        name = simpledialog.askstring(
            "Preset de escrita", "Nome do preset:",
            initialvalue=f"{suggested_preset_name(settings)}_var-{index}", parent=self,
        )
        if not name:
            return
        preset = upsert_preset({
            "uid": new_preset_uid("style"),
            "name": name.strip(),
            "created_from": "handwriting_gallery",
            "settings": settings,
        })
        self.status_var.set(f"Preset salvo: {preset['uid']}")

    def generate_gallery(self) -> None:
        try:
            variations = self._variations()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Galeria de escrita", str(exc))
            return
        self.generate_button.configure(state="disabled")
        self.status_var.set("Gerando amostras...")
        model = self.model_var.get()
        writer_text = self.writer_var.get()
        writer = None if writer_text == "Aleatório" else int(writer_text)
        threading.Thread(
            target=self._worker,
            args=(model, writer, float(self.bias_var.get()), self.text_policy_var.get(), variations),
            daemon=True,
        ).start()

    def _worker(self, model_name: str, writer: int | None, bias: float,
                text_policy: str, variations: list[GalleryVariation]) -> None:
        try:
            rows = render_gallery_samples(model_name, writer, bias, text_policy, variations)
            self.events.put(("done", rows))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self.generate_button.configure(state="normal")
                if kind == "done":
                    self._show_gallery(payload)
                    self.status_var.set("Galeria pronta.")
                else:
                    self.status_var.set("Falha ao gerar galeria.")
                    messagebox.showerror("Galeria de escrita", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _show_gallery(self, rows: list[dict[str, object]]) -> None:
        for child in self.gallery_frame.winfo_children():
            child.destroy()
        self.gallery_images.clear()
        headings = ("Campo", "Texto", "Variação 1", "Variação 2", "Variação 3")
        for column, heading in enumerate(headings):
            ttk.Label(self.gallery_frame, text=heading, style="TLabel", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=column, sticky="w", padx=8, pady=6)
        for row_index, row in enumerate(rows, start=1):
            ttk.Label(self.gallery_frame, text=str(row["label"]), style="TLabel").grid(row=row_index, column=0, sticky="w", padx=8, pady=6)
            ttk.Label(self.gallery_frame, text=str(row["text"]), style="TLabel").grid(row=row_index, column=1, sticky="w", padx=8, pady=6)
            for column, image in enumerate(row["images"], start=2):
                photo = ImageTk.PhotoImage(image)
                self.gallery_images.append(photo)
                label = ttk.Label(self.gallery_frame, image=photo, background="#ffffff")
                label.grid(row=row_index, column=column, padx=8, pady=6)


def render_gallery_samples(model_name: str, writer: int | None, bias: float,
                           text_policy: str, variations: list[GalleryVariation]) -> list[dict[str, object]]:
    rng = random.Random(42)
    handwriting_model = None if model_name == "font" else load_model(model_name)
    if model_name != "font" and writer is None:
        writer = rng.randrange(13 if model_name == "sjvasquez" else 512)
    rows = []
    for label, text in SAMPLE_TEXTS:
        images = []
        for index, variation in enumerate(variations):
            seed = 10_000 + index * 100 + len(text)
            if model_name == "font":
                style = random_handwriting_style(random.Random(seed), intensity=0.75, medium="pen")
                image = (
                    render_handwritten_numeric(text, 260, 58, random.Random(seed), style=style, intensity=0.75)
                    if all(character in "0123456789.,:/- " for character in text)
                    else render_handwritten_text(text, 260, 58, random.Random(seed), style=style, intensity=0.75, medium="pen")
                )
                if variation.color != "#193787":
                    image = _recolor(image, variation.color)
            else:
                pen = NeuralPen(
                    model=model_name, writer=writer, bias=bias, color=variation.color,
                    width=variation.width, pressure=variation.pressure,
                    dropout=variation.dropout, text_policy=text_policy,
                )
                prepared = prepare_text(text, text_policy)
                points = handwriting_model.sample(prepared, pen.writer or 0, pen.bias, seed)
                image = render_strokes(points, 260, 58, pen, seed)
            images.append(_on_white(image))
        rows.append({"label": label, "text": text, "images": images})
    return rows


def _recolor(image: Image.Image, color: str) -> Image.Image:
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    result = Image.new("RGBA", image.size, (red, green, blue, 0))
    result.putalpha(image.getchannel("A"))
    return result


def _on_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A"))
    stream = io.BytesIO()
    background.save(stream, format="PNG")
    stream.seek(0)
    return Image.open(stream).copy()
