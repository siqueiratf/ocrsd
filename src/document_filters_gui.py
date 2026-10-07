from __future__ import annotations

from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.document_filters import apply_document_filters
from src.pdf import pdf_to_image


class DocumentFiltersFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.pdf_path: Path | None = None
        self.original_image = None
        self.filtered_image = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preset_var = tk.StringVar(value="medium")
        self.status_var = tk.StringVar(value="Abra um PDF gerado para calibrar o realismo visual.")
        self._rng = random.Random()
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Label(toolbar, text="Realismo visual", style="Panel.TLabel", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Abrir PDF…", command=self.open_pdf).pack(side="left", padx=(14, 4))
        ttk.Button(toolbar, text="Randomizar preview", command=self.randomize_preview).pack(side="left", padx=4)
        ttk.Label(toolbar, text="Preset", style="Panel.TLabel").pack(side="left", padx=(18, 5))
        preset_box = ttk.Combobox(
            toolbar, textvariable=self.preset_var,
            values=("light", "medium", "strong"), state="readonly", width=10,
        )
        preset_box.pack(side="left")
        preset_box.bind("<<ComboboxSelected>>", lambda _event: self.randomize_preview())
        ttk.Label(toolbar, textvariable=self.status_var, style="Panel.TLabel").pack(side="left", fill="x", expand=True, padx=12)

        self.canvas = tk.Canvas(self, bg="#0e1012", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.render())

    def open_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        try:
            self.load_pdf(Path(path))
        except Exception as exc:
            messagebox.showerror("Realismo visual", str(exc))

    def load_pdf(self, path: Path, dpi: int = 120) -> None:
        self.pdf_path = path
        self.original_image = pdf_to_image(str(path), dpi).image
        self.randomize_preview()

    def randomize_preview(self) -> None:
        if self.original_image is None:
            self.render()
            return
        seed = self._rng.getrandbits(64)
        self.filtered_image, metadata = apply_document_filters(
            self.original_image, rng=random.Random(seed), preset=self.preset_var.get(),
        )
        self.status_var.set(
            f"{self.pdf_path.name if self.pdf_path else 'preview'} · "
            f"{metadata['preset']} · contraste {metadata['contrast']:.2f} · "
            f"ruído {metadata['noise_sigma']:.1f}"
        )
        self.render()

    def render(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None or self.filtered_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                text="Abra um PDF para visualizar original e filtrado lado a lado.",
                fill="#a9afb8",
            )
            return
        original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        filtered = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2RGB)
        gap = 24
        height = max(original.shape[0], filtered.shape[0])
        width = original.shape[1] + filtered.shape[1] + gap
        combined = Image.new("RGB", (width, height), "#0e1012")
        combined.paste(Image.fromarray(original), (0, 0))
        combined.paste(Image.fromarray(filtered), (original.shape[1] + gap, 0))

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = min(canvas_width / combined.width, canvas_height / combined.height, 1.0)
        render_size = (max(1, round(combined.width * scale)), max(1, round(combined.height * scale)))
        if render_size != combined.size:
            combined = combined.resize(render_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(combined)
        self.canvas.create_image(
            (canvas_width - render_size[0]) // 2,
            (canvas_height - render_size[1]) // 2,
            anchor="nw", image=self.preview_photo,
        )
        self.canvas.create_text(18, 18, anchor="nw", text="Original", fill="#f2f3f5")
        midpoint = (canvas_width - render_size[0]) // 2 + round((original.shape[1] + gap) * scale)
        self.canvas.create_text(midpoint + 18, 18, anchor="nw", text="Filtrado", fill="#f2f3f5")
