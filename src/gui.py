from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.ocr_bench import BACKEND_FACTORIES
from src.gui_common import disable_wheel_value_changes
from src.field_mapper_gui import FieldMapperFrame
from src.generated_pdf_viewer_gui import GeneratedPdfViewerFrame
from src.handwriting_gallery_gui import HandwritingGalleryFrame
from src.document_filters_gui import DocumentFiltersFrame
from src.pipeline_gui import PipelineRunnerFrame
from src.synthetic_dataset_gui import SyntheticDatasetFrame


BG = "#15171a"
PANEL = "#202328"
PANEL_2 = "#292d33"
TEXT = "#f2f3f5"
MUTED = "#a9afb8"
ACCENT = "#7c9cff"
ACCENT_ACTIVE = "#6688ef"
ERROR = "#ff6b6b"


def collect_folder_images(folder: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        item for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in extensions
    )


class OcrDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RCSD · Imagens e testes OCR")
        self.geometry("1120x720")
        self.minsize(880, 600)
        self.configure(bg=BG)

        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.history: list[np.ndarray] = []
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.crop_start: tuple[int, int] | None = None
        self.crop_rect: int | None = None
        self.crop_box: tuple[int, int, int, int] | None = None
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.test_images: list[Path] = []
        self.test_folder: Path | None = None
        self.test_index = 0
        self.test_photo: ImageTk.PhotoImage | None = None
        self.run_id = 0
        self.run_started_at = 0.0
        self.batch_rows: list[dict[str, object]] = []
        self.sort_reverse = {"confidence": True, "latency": False}
        self.hue_min_var = tk.DoubleVar(value=80)
        self.hue_max_var = tk.DoubleVar(value=140)
        self.sat_min_var = tk.DoubleVar(value=10)
        self.gray_max_var = tk.DoubleVar(value=220)
        self.ink_threshold_var = tk.DoubleVar(value=250)
        self.noise_area_var = tk.DoubleVar(value=4)
        self.close_kernel_var = tk.DoubleVar(value=3)
        self.background_threshold_var = tk.DoubleVar(value=245)
        self.preprocess_preview_source: np.ndarray | None = None
        self.active_preprocess_step: str | None = None
        self.preprocess_preview_job: str | None = None
        self.parameter_groups: dict[str, list[tuple[ttk.Frame, ttk.Label, ttk.Label, ttk.Scale]]] = {}

        self._configure_style()
        disable_wheel_value_changes(self)
        self._build_ui()
        self.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL_2)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("TButton", background=PANEL_2, foreground=TEXT, padding=(12, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#363b43"), ("disabled", PANEL)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#0d1020")
        style.map("Accent.TButton", background=[("active", ACCENT_ACTIVE)])
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL_2)], foreground=[("selected", TEXT)])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=28)
        style.configure("Treeview.Heading", background=PANEL_2, foreground=TEXT)
        style.map("Treeview", background=[("selected", "#34446f")])
        style.configure("TCombobox", fieldbackground=PANEL_2, foreground=TEXT)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL_2)
        style.configure("Parameter.TFrame", background=PANEL)
        style.configure("ActiveParameter.TFrame", background="#34446f")
        style.configure("Parameter.TLabel", background=PANEL, foreground=TEXT)
        style.configure("ActiveParameter.TLabel", background="#34446f", foreground=TEXT)
        style.configure("Parameter.Horizontal.TScale", background=PANEL, troughcolor=PANEL_2)
        style.configure("ActiveParameter.Horizontal.TScale", background="#34446f", troughcolor=ACCENT)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(20, 16, 20, 8))
        header.pack(fill="x")
        ttk.Label(header, text="RCSD", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="Preparação de imagens e comparação OCR", style="Muted.TLabel").pack(side="left", padx=14)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        editor = ttk.Frame(tabs)
        benchmark = ttk.Frame(tabs)
        field_mapper = FieldMapperFrame(tabs)
        pipeline_runner = PipelineRunnerFrame(tabs)
        synthetic_dataset = SyntheticDatasetFrame(tabs)
        generated_pdf_viewer = GeneratedPdfViewerFrame(tabs)
        synthetic_dataset.set_generated_pdf_viewer(generated_pdf_viewer, tabs)
        handwriting_gallery = HandwritingGalleryFrame(tabs)
        document_filters = DocumentFiltersFrame(tabs)
        tabs.add(editor, text="Editor de imagem")
        tabs.add(field_mapper, text="Mapeador de campos")
        tabs.add(pipeline_runner, text="Executar documento")
        tabs.add(synthetic_dataset, text="Datasets artificiais")
        tabs.add(generated_pdf_viewer, text="PDFs gerados")
        tabs.add(handwriting_gallery, text="Galeria de escrita")
        tabs.add(document_filters, text="Realismo visual")
        tabs.add(benchmark, text="Testes de modelos")
        self._build_editor(editor)
        self._build_benchmark(benchmark)

    def _build_editor(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", pady=(10, 8))
        for label, command in (
            ("Abrir", self.open_image), ("Girar ↶", lambda: self.rotate(90)),
            ("Girar ↷", lambda: self.rotate(-90)), ("Cinza", self.grayscale),
            ("Todas as etapas", lambda: self.start_preprocess_preview("all")),
            ("VLM", lambda: self.preprocess("vlm")), ("Recortar", self.apply_crop),
            ("Desfazer", self.undo), ("Restaurar", self.restore),
        ):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Salvar como…", style="Accent.TButton", command=self.save_as).pack(side="right", padx=3)
        ttk.Button(toolbar, text="Salvar", command=self.save).pack(side="right", padx=3)

        stages = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 8))
        stages.pack(fill="x", pady=(0, 8))
        stage_buttons = ttk.Frame(stages, style="Panel.TFrame")
        stage_buttons.pack(fill="x")
        ttk.Label(stage_buttons, text="Etapas:", style="Panel.TLabel").pack(side="left", padx=(0, 6))
        for label, step in (
            ("1 · Remover linhas", "lines"),
            ("2 · Azul → preto", "black"),
            ("3 · Remover ruído", "noise"),
            ("4 · Fundo branco", "background"),
        ):
            ttk.Button(
                stage_buttons, text=label,
                command=lambda selected=step: self.start_preprocess_preview(selected),
            ).pack(side="left", padx=3)
        self.active_stage_label = ttk.Label(
            stage_buttons, text="Clique numa etapa para ativar a prévia automática.",
            style="Panel.TLabel", foreground=MUTED,
        )
        self.active_stage_label.pack(side="left", padx=12)

        parameters = ttk.Frame(stages, style="Panel.TFrame")
        parameters.pack(fill="x", pady=(8, 0))

        def parameter(row: ttk.Frame, stage_key: str, label: str, variable: tk.DoubleVar,
                      start: int, end: int) -> None:
            box = ttk.Frame(row, style="Parameter.TFrame", padding=(4, 2))
            box.pack(side="left", fill="x", expand=True, padx=(4, 8))
            value_label = ttk.Label(box, text=str(round(variable.get())), style="Parameter.TLabel", width=4, anchor="e")
            name_label = ttk.Label(box, text=label, style="Parameter.TLabel")
            name_label.pack(side="left")
            value_label.pack(side="right")
            variable.trace_add(
                "write",
                lambda *_args, shown=value_label, current=variable:
                    shown.configure(text=str(round(current.get()))),
            )
            slider = ttk.Scale(
                box, variable=variable, from_=start, to=end,
                command=lambda value, shown=value_label: self._slider_changed(shown, value),
                style="Parameter.Horizontal.TScale",
            )
            slider.pack(side="right", fill="x", expand=True, padx=5)
            self.parameter_groups.setdefault(stage_key, []).append((box, name_label, value_label, slider))

        def parameter_row(stage: str) -> ttk.Frame:
            row = ttk.Frame(parameters, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=stage, style="Panel.TLabel", width=16).pack(side="left")
            return row

        row = parameter_row("1 · Remover linhas")
        parameter(row, "lines", "Azul mín.", self.hue_min_var, 0, 179)
        parameter(row, "lines", "Azul máx.", self.hue_max_var, 0, 179)
        parameter(row, "lines", "Saturação", self.sat_min_var, 0, 255)
        parameter(row, "lines", "Escuro", self.gray_max_var, 0, 255)
        row = parameter_row("2 · Azul → preto")
        parameter(row, "black", "Tolerância da tinta", self.ink_threshold_var, 0, 255)
        row = parameter_row("3 · Remover ruído")
        parameter(row, "noise", "Área mínima (px)", self.noise_area_var, 1, 500)
        parameter(row, "noise", "Fechamento", self.close_kernel_var, 1, 15)
        row = parameter_row("4 · Fundo branco")
        parameter(row, "background", "Tolerância do fundo", self.background_threshold_var, 0, 255)

        preset_actions = ttk.Frame(stages, style="Panel.TFrame")
        preset_actions.pack(fill="x", pady=(7, 0))
        ttk.Label(preset_actions, text="Preset:", style="Panel.TLabel", width=16).pack(side="left")
        ttk.Button(preset_actions, text="Salvar preset", command=self.save_preprocess_preset).pack(side="left", padx=3)
        ttk.Button(preset_actions, text="Carregar preset", command=self.load_preprocess_preset).pack(side="left", padx=3)
        self.folder_preprocess_button = ttk.Button(
            preset_actions, text="Aplicar preset em uma pasta", style="Accent.TButton",
            command=self.apply_preset_to_folder,
        )
        self.folder_preprocess_button.pack(side="left", padx=3)

        self.canvas = tk.Canvas(parent, bg="#0e1012", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.render_image())
        self.canvas.bind("<ButtonPress-1>", self._crop_press)
        self.canvas.bind("<B1-Motion>", self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_release)
        self.editor_status = ttk.Label(parent, text="Abra uma imagem para começar. Arraste sobre ela para selecionar um recorte.", style="Muted.TLabel")
        self.editor_status.pack(fill="x", pady=(8, 0))

    def _build_benchmark(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Button(toolbar, text="Selecionar pasta", style="Accent.TButton", command=self.choose_test_folder).pack(side="left")
        self.input_var = tk.StringVar(value="Nenhuma pasta selecionada")
        ttk.Label(toolbar, textvariable=self.input_var, style="Panel.TLabel").pack(side="left", padx=12)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        viewer = ttk.Frame(body, style="Panel.TFrame", padding=12)
        viewer.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.test_canvas = tk.Canvas(viewer, bg="#0e1012", highlightthickness=0)
        self.test_canvas.pack(fill="both", expand=True)
        self.test_canvas.bind("<Configure>", lambda _e: self.render_test_image())
        navigation = ttk.Frame(viewer, style="Panel.TFrame")
        navigation.pack(fill="x", pady=(10, 0))
        self.prev_button = ttk.Button(navigation, text="<", width=5, command=lambda: self.navigate_test_image(-1), state="disabled")
        self.prev_button.pack(side="left")
        self.test_image_label = ttk.Label(navigation, text="Selecione uma pasta com imagens", style="Panel.TLabel", anchor="center")
        self.test_image_label.pack(side="left", fill="x", expand=True, padx=10)
        self.next_button = ttk.Button(navigation, text=">", width=5, command=lambda: self.navigate_test_image(1), state="disabled")
        self.next_button.pack(side="right")

        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=16, width=330)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Modelo OCR", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.model_var = tk.StringVar(value="easyocr")
        ttk.Combobox(sidebar, textvariable=self.model_var, state="readonly", values=tuple(BACKEND_FACTORIES)).pack(fill="x", pady=(7, 0))

        ttk.Label(sidebar, text="Pré-processamento", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(18, 6))
        self.preprocess_var = tk.StringVar(value="none")
        ttk.Combobox(sidebar, textvariable=self.preprocess_var, state="readonly", values=("none", "traditional", "vlm")).pack(fill="x")
        self.run_button = ttk.Button(sidebar, text="Reconhecer imagem", style="Accent.TButton", command=self.run_benchmark)
        self.run_button.pack(fill="x", pady=(22, 6))
        self.batch_button = ttk.Button(sidebar, text="Testar pasta inteira", command=self.run_batch)
        self.batch_button.pack(fill="x", pady=(0, 6))
        self.cancel_button = ttk.Button(sidebar, text="Cancelar", command=self.cancel_benchmark, state="disabled")
        self.cancel_button.pack(fill="x")
        self.progress = ttk.Progressbar(sidebar, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))
        ttk.Label(sidebar, text="Resultado", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(12, 6))
        self.result_text = tk.Text(sidebar, height=5, wrap="word", bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="flat", padx=10, pady=10, state="disabled")
        self.result_text.pack(fill="x")
        self.result_meta = ttk.Label(sidebar, text="", style="Panel.TLabel", foreground=MUTED, wraplength=290)
        self.result_meta.pack(fill="x", pady=(8, 0))
        ttk.Label(sidebar, text="Resultados da pasta", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(14, 6))
        self.batch_results = ttk.Treeview(
            sidebar, columns=("image", "confidence", "latency"), show="headings", height=8
        )
        self.batch_results.heading("image", text="Imagem")
        self.batch_results.heading(
            "confidence", text="Certeza ↕",
            command=lambda: self.sort_batch_results("confidence"),
        )
        self.batch_results.heading(
            "latency", text="Tempo ↕",
            command=lambda: self.sort_batch_results("latency"),
        )
        self.batch_results.column("image", width=150, minwidth=80)
        self.batch_results.column("confidence", width=72, anchor="center")
        self.batch_results.column("latency", width=72, anchor="center")
        self.batch_results.pack(fill="both", expand=True)
        self.batch_results.bind("<Double-1>", self.open_batch_result)
        self.bench_status = ttk.Label(sidebar, text="Aguardando uma imagem.", style="Panel.TLabel", wraplength=290)
        self.bench_status.pack(fill="x", side="bottom")


    def open_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")])
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Imagem inválida", "Não foi possível abrir essa imagem.")
            return
        self.image_path = Path(path)
        self.original_image = image.copy()
        self.current_image = image
        self.history.clear()
        self.active_preprocess_step = None
        self.preprocess_preview_source = None
        if self.preprocess_preview_job is not None:
            self.after_cancel(self.preprocess_preview_job)
            self.preprocess_preview_job = None
        self._highlight_parameter_groups(None)
        self.crop_box = None
        self.render_image()

    def _remember(self) -> bool:
        if self.current_image is None:
            messagebox.showinfo("Editor", "Abra uma imagem primeiro.")
            return False
        self.active_preprocess_step = None
        self.preprocess_preview_source = None
        if self.preprocess_preview_job is not None:
            self.after_cancel(self.preprocess_preview_job)
            self.preprocess_preview_job = None
        self._highlight_parameter_groups(None)
        self.history.append(self.current_image.copy())
        if len(self.history) > 20:
            self.history.pop(0)
        return True

    def rotate(self, degrees: int) -> None:
        if not self._remember(): return
        code = cv2.ROTATE_90_COUNTERCLOCKWISE if degrees > 0 else cv2.ROTATE_90_CLOCKWISE
        self.current_image = cv2.rotate(self.current_image, code)
        self.crop_box = None
        self.render_image()

    def grayscale(self) -> None:
        if not self._remember(): return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.render_image()

    def preprocess(self, mode: str) -> None:
        if not self._remember(): return
        try:
            from src.preprocess import preprocess_for_ocr
            result = preprocess_for_ocr(
                self.current_image, mode=mode,
                hue_range=(self._parameter(self.hue_min_var), self._parameter(self.hue_max_var)),
                sat_min=self._parameter(self.sat_min_var), gray_max=self._parameter(self.gray_max_var),
                min_component_area=self._parameter(self.noise_area_var),
                close_kernel=self._parameter(self.close_kernel_var),
                ink_threshold=self._parameter(self.ink_threshold_var),
                background_threshold=self._parameter(self.background_threshold_var),
            )
            self.current_image = result if result.ndim == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.render_image()
        except Exception as exc:
            self.current_image = self.history.pop()
            messagebox.showerror("Pré-processamento", str(exc))

    @staticmethod
    def _parameter(variable: tk.DoubleVar) -> int:
        return round(variable.get())

    def _slider_changed(self, value_label: ttk.Label, value: str) -> None:
        value_label.configure(text=str(round(float(value))))
        if self.active_preprocess_step is None:
            return
        if self.preprocess_preview_job is not None:
            self.after_cancel(self.preprocess_preview_job)
        self.preprocess_preview_job = self.after(120, self._render_preprocess_preview)

    def _highlight_parameter_groups(self, active_step: str | None) -> None:
        active_groups = set(self.parameter_groups) if active_step == "all" else {active_step}
        for stage, widgets in self.parameter_groups.items():
            active = stage in active_groups
            frame_style = "ActiveParameter.TFrame" if active else "Parameter.TFrame"
            label_style = "ActiveParameter.TLabel" if active else "Parameter.TLabel"
            scale_style = "ActiveParameter.Horizontal.TScale" if active else "Parameter.Horizontal.TScale"
            for box, name_label, value_label, slider in widgets:
                box.configure(style=frame_style)
                name_label.configure(style=label_style)
                value_label.configure(style=label_style)
                slider.configure(style=scale_style)

    def start_preprocess_preview(self, step: str) -> None:
        if self.current_image is None:
            messagebox.showinfo("Editor", "Abra uma imagem primeiro.")
            return
        source = self.current_image.copy()
        if not self._remember():
            return
        self.preprocess_preview_source = source
        self.active_preprocess_step = step
        self._highlight_parameter_groups(step)
        names = {
            "lines": "1 · Remover linhas", "black": "2 · Azul → preto",
            "noise": "3 · Remover ruído", "background": "4 · Fundo branco",
            "all": "Todas as etapas",
        }
        self.active_stage_label.configure(text=f"Prévia ativa: {names[step]}. Mova os sliders para comparar.")
        self._render_preprocess_preview()

    def _render_preprocess_preview(self) -> None:
        self.preprocess_preview_job = None
        step = self.active_preprocess_step
        source = self.preprocess_preview_source
        if step is None or source is None:
            return
        try:
            from src.preprocess import (
                convert_ink_to_black, preprocess_for_ocr, remove_form_lines,
                remove_ink_noise, whiten_background,
            )
            if step == "lines":
                result = remove_form_lines(
                    source,
                    hue_range=(self._parameter(self.hue_min_var), self._parameter(self.hue_max_var)),
                    sat_min=self._parameter(self.sat_min_var), gray_max=self._parameter(self.gray_max_var),
                )
                label = "Linhas pretas removidas"
            elif step == "black":
                result = convert_ink_to_black(source, self._parameter(self.ink_threshold_var))
                label = "Tinta convertida para preto"
            elif step == "noise":
                result = remove_ink_noise(
                    source, self._parameter(self.noise_area_var), self._parameter(self.close_kernel_var)
                )
                label = "Ruído removido"
            elif step == "background":
                result = whiten_background(source, self._parameter(self.background_threshold_var))
                label = "Fundo convertido para branco"
            elif step == "all":
                result = preprocess_for_ocr(
                    source, mode="traditional",
                    hue_range=(self._parameter(self.hue_min_var), self._parameter(self.hue_max_var)),
                    sat_min=self._parameter(self.sat_min_var), gray_max=self._parameter(self.gray_max_var),
                    min_component_area=self._parameter(self.noise_area_var),
                    close_kernel=self._parameter(self.close_kernel_var),
                    ink_threshold=self._parameter(self.ink_threshold_var),
                    background_threshold=self._parameter(self.background_threshold_var),
                )
                label = "Todas as etapas aplicadas"
            else:
                raise ValueError(f"Etapa desconhecida: {step}")
            self.current_image = (
                result if result.ndim == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            )
            self.crop_box = None
            self.render_image()
            self.editor_status.configure(text=f"{label}. A prévia será atualizada ao mover um slider.")
        except Exception as exc:
            self.current_image = source
            self.active_preprocess_step = None
            self.preprocess_preview_source = None
            self._highlight_parameter_groups(None)
            messagebox.showerror("Pré-processamento", str(exc))

    def _preprocess_parameters(self) -> dict[str, int | list[int]]:
        return {
            "hue_range": [self._parameter(self.hue_min_var), self._parameter(self.hue_max_var)],
            "sat_min": self._parameter(self.sat_min_var),
            "gray_max": self._parameter(self.gray_max_var),
            "ink_threshold": self._parameter(self.ink_threshold_var),
            "min_component_area": self._parameter(self.noise_area_var),
            "close_kernel": self._parameter(self.close_kernel_var),
            "background_threshold": self._parameter(self.background_threshold_var),
        }

    def save_preprocess_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="preset_preprocessamento.json",
            filetypes=[("Preset JSON", "*.json")],
        )
        if not path:
            return
        preset = {"preset_version": 1, "mode": "traditional", "parameters": self._preprocess_parameters()}
        try:
            Path(path).write_text(json.dumps(preset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.editor_status.configure(text=f"Preset salvo em {path}")
        except Exception as exc:
            messagebox.showerror("Preset", f"Não foi possível salvar o preset: {exc}")

    def load_preprocess_preset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Preset JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("preset_version") != 1 or data.get("mode") != "traditional":
                raise ValueError("Formato ou versão de preset incompatível")
            values = data["parameters"]
            hue_range = values["hue_range"]
            assignments = (
                (self.hue_min_var, hue_range[0], 0, 179),
                (self.hue_max_var, hue_range[1], 0, 179),
                (self.sat_min_var, values["sat_min"], 0, 255),
                (self.gray_max_var, values["gray_max"], 0, 255),
                (self.ink_threshold_var, values["ink_threshold"], 0, 255),
                (self.noise_area_var, values["min_component_area"], 1, 500),
                (self.close_kernel_var, values["close_kernel"], 1, 15),
                (self.background_threshold_var, values["background_threshold"], 0, 255),
            )
            for variable, value, minimum, maximum in assignments:
                numeric = int(value)
                if not minimum <= numeric <= maximum:
                    raise ValueError(f"Parâmetro fora do intervalo: {numeric}")
                variable.set(numeric)
            self.editor_status.configure(text=f"Preset carregado de {path}")
            if self.active_preprocess_step is not None:
                self._render_preprocess_preview()
        except Exception as exc:
            messagebox.showerror("Preset", f"Não foi possível carregar o preset: {exc}")

    def apply_preset_to_folder(self) -> None:
        source = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if not source:
            return
        destination = filedialog.askdirectory(title="Selecione uma pasta diferente para os resultados")
        if not destination:
            return
        source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
        if source_path == destination_path:
            messagebox.showerror("Processar pasta", "Escolha uma pasta de saída diferente para não sobrescrever os originais.")
            return
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted(
            path for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
            and destination_path not in path.resolve().parents
        )
        if not images:
            messagebox.showinfo("Processar pasta", "Nenhuma imagem compatível foi encontrada.")
            return
        parameters = self._preprocess_parameters()
        self.folder_preprocess_button.configure(state="disabled")
        self.active_stage_label.configure(text=f"Processando 0 de {len(images)} imagens…")
        threading.Thread(
            target=self._folder_preprocess_worker,
            args=(source_path, destination_path, images, parameters), daemon=True,
        ).start()

    def _folder_preprocess_worker(self, source: Path, destination: Path,
                                  images: list[Path], parameters: dict) -> None:
        try:
            from src.preprocess import preprocess_for_ocr
            for index, image_path in enumerate(images, start=1):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Não foi possível abrir {image_path}")
                result = preprocess_for_ocr(image, mode="traditional", **parameters)
                output = destination / image_path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output), result):
                    raise OSError(f"Não foi possível salvar {output}")
                self.events.put(("preprocess_progress", (index, len(images), image_path.name)))
            self.events.put(("preprocess_folder_done", (len(images), destination)))
        except Exception as exc:
            self.events.put(("preprocess_folder_error", str(exc)))

    def undo(self) -> None:
        if self.history:
            self.active_preprocess_step = None
            self.preprocess_preview_source = None
            if self.preprocess_preview_job is not None:
                self.after_cancel(self.preprocess_preview_job)
                self.preprocess_preview_job = None
            self._highlight_parameter_groups(None)
            self.current_image = self.history.pop()
            self.crop_box = None
            self.render_image()
            self.active_stage_label.configure(text="Prévia encerrada. Selecione outra etapa para continuar.")

    def restore(self) -> None:
        if self.original_image is not None and self._remember():
            self.current_image = self.original_image.copy()
            self.crop_box = None
            self.render_image()

    def render_image(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None:
            self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, text="Nenhuma imagem aberta", fill=MUTED, font=("TkDefaultFont", 14))
            return
        h, w = self.current_image.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.preview_scale = min(cw / w, ch / h, 1.0)
        rw, rh = max(1, int(w * self.preview_scale)), max(1, int(h * self.preview_scale))
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA) if (rw, rh) != (w, h) else rgb
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(resized))
        ox, oy = (cw - rw) // 2, (ch - rh) // 2
        self.preview_offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor="nw", image=self.preview_photo)
        name = self.image_path.name if self.image_path else "imagem"
        self.editor_status.configure(text=f"{name} · {w} × {h} px · arraste para selecionar uma área")

    def _crop_press(self, event: tk.Event) -> None:
        if self.current_image is None: return
        self.crop_start = (event.x, event.y)
        if self.crop_rect: self.canvas.delete(self.crop_rect)
        self.crop_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=ACCENT, width=2)

    def _crop_drag(self, event: tk.Event) -> None:
        if self.crop_start and self.crop_rect:
            self.canvas.coords(self.crop_rect, self.crop_start[0], self.crop_start[1], event.x, event.y)

    def _crop_release(self, event: tk.Event) -> None:
        if not self.crop_start or self.current_image is None: return
        ox, oy = self.preview_offset
        h, w = self.current_image.shape[:2]
        x1, x2 = sorted((self.crop_start[0], event.x)); y1, y2 = sorted((self.crop_start[1], event.y))
        coords = (int((x1-ox)/self.preview_scale), int((y1-oy)/self.preview_scale), int((x2-ox)/self.preview_scale), int((y2-oy)/self.preview_scale))
        ax, ay, bx, by = max(0, coords[0]), max(0, coords[1]), min(w, coords[2]), min(h, coords[3])
        self.crop_box = (ax, ay, bx, by) if bx-ax > 2 and by-ay > 2 else None
        self.crop_start = None

    def apply_crop(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("Recorte", "Abra uma imagem primeiro.")
        elif not self.crop_box:
            messagebox.showinfo("Recorte", "Arraste sobre a imagem para selecionar a área.")
        elif self._remember():
            x1, y1, x2, y2 = self.crop_box
            self.current_image = self.current_image[y1:y2, x1:x2].copy()
            self.crop_box = None
            self.render_image()

    def save(self) -> None:
        if self.current_image is None or self.image_path is None: return
        if not messagebox.askyesno("Salvar", f"Substituir {self.image_path.name}?"):
            return
        self._write_image(self.image_path)

    def save_as(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("Salvar", "Abra uma imagem primeiro.")
            return
        initial = self.image_path.stem + "_editada.png" if self.image_path else "imagem_editada.png"
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=initial, filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")])
        if path: self._write_image(Path(path))

    def _write_image(self, path: Path) -> None:
        if not cv2.imwrite(str(path), self.current_image):
            messagebox.showerror("Salvar", "Não foi possível salvar a imagem.")
        else:
            self.image_path = path
            self.editor_status.configure(text=f"Salvo em {path}")


    def choose_test_folder(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return
        self.test_folder = Path(path)
        self.test_images = collect_folder_images(self.test_folder)
        self.input_var.set(path)
        self.test_index = 0
        self._clear_test_result()
        if not self.test_images:
            self.test_canvas.delete("all")
            self.test_image_label.configure(text="Nenhuma imagem encontrada")
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            self.bench_status.configure(text="A pasta não contém imagens compatíveis.")
            return
        self.render_test_image()

    def render_test_image(self) -> None:
        self.test_canvas.delete("all")
        if not self.test_images:
            self.test_canvas.create_text(
                self.test_canvas.winfo_width() / 2,
                self.test_canvas.winfo_height() / 2,
                text="Selecione uma pasta",
                fill=MUTED,
                font=("TkDefaultFont", 14),
            )
            return
        path = self.test_images[self.test_index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.test_canvas.create_text(20, 20, anchor="nw", text="Não foi possível abrir a imagem", fill=ERROR)
            return
        h, w = image.shape[:2]
        cw, ch = max(1, self.test_canvas.winfo_width()), max(1, self.test_canvas.winfo_height())
        scale = min(cw / w, ch / h, 1.0)
        rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (rw, rh) != (w, h):
            rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        self.test_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.test_canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.test_photo)
        self.test_image_label.configure(
            text=f"{self.test_index + 1} de {len(self.test_images)}  ·  {path.name}"
        )
        state = "normal" if len(self.test_images) > 1 else "disabled"
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)
        self.bench_status.configure(text="Escolha o modelo e clique em Reconhecer imagem.")

    def navigate_test_image(self, step: int) -> None:
        if not self.test_images:
            return
        if self.process is not None:
            self.cancel_benchmark()
        self.test_index = (self.test_index + step) % len(self.test_images)
        self._clear_test_result()
        self.render_test_image()

    def _clear_test_result(self) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.configure(state="disabled")
        self.result_meta.configure(text="")

    def run_benchmark(self) -> None:
        if not self.test_images:
            messagebox.showerror("Entrada", "Selecione uma pasta com imagens.")
            return
        image_path = self.test_images[self.test_index]
        self._start_benchmark(image_path, "single")

    def run_batch(self) -> None:
        if not self.test_images:
            messagebox.showerror("Entrada", "Selecione uma pasta com imagens.")
            return
        assert self.test_folder is not None
        self._start_benchmark(self.test_folder, "batch")

    def _start_benchmark(self, input_path: Path, mode: str) -> None:
        model = self.model_var.get()
        suffix = "current" if mode == "single" else "batch"
        output = Path.cwd() / f".ocr_comparison_gui_{suffix}.json"
        output.unlink(missing_ok=True)
        self.run_id += 1
        current_run = self.run_id
        command = [
            sys.executable, "-u", "-m", "src.ocr_bench",
            "--input", str(input_path), "--models", model,
            "--preprocess", self.preprocess_var.get(), "--output", str(output),
        ]
        self._clear_test_result()
        if mode == "batch":
            self.batch_rows.clear()
            for item in self.batch_results.get_children():
                self.batch_results.delete(item)
        self.run_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.run_started_at = time.monotonic()
        action = "pasta inteira" if mode == "batch" else input_path.name
        self.bench_status.configure(text=f"Carregando {model} para {action}…")
        threading.Thread(target=self._benchmark_worker, args=(command, output, current_run, mode), daemon=True).start()
        timeout_seconds = 60 if mode == "single" else max(60, len(self.test_images) * 30)
        self.after(
            timeout_seconds * 1000,
            lambda: self._check_benchmark_timeout(current_run, model, timeout_seconds),
        )

    def _benchmark_worker(self, command: list[str], output: Path, run_id: int, mode: str) -> None:
        local_process: subprocess.Popen[str] | None = None
        try:
            local_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.process = local_process
            assert local_process.stdout is not None
            for line in local_process.stdout:
                self.events.put(("log", (run_id, line.strip())))
            code = local_process.wait()
            self.events.put(("done", (run_id, code, output, mode)))
        except Exception as exc:
            self.events.put(("error", (run_id, str(exc))))
        finally:
            if self.process is local_process:
                self.process = None

    def _check_benchmark_timeout(self, run_id: int, model: str, timeout_seconds: int) -> None:
        if run_id != self.run_id or self.process is None:
            return
        self.run_id += 1
        self.process.terminate()
        self._set_idle()
        self.bench_status.configure(text=f"{model} excedeu {timeout_seconds} segundos e foi interrompido.")
        self.result_meta.configure(text="Verifique se os arquivos do modelo estão instalados. O EasyOCR não ficará mais carregando indefinidamente.")

    def cancel_benchmark(self) -> None:
        if self.process is not None:
            self.run_id += 1
            self.process.terminate()
            self.bench_status.configure(text="Cancelando…")
            self._set_idle()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log" and payload:
                    run_id, line = payload
                    if run_id == self.run_id and line:
                        self.bench_status.configure(text=str(line)[-130:])
                elif kind == "done":
                    run_id, code, output, mode = payload
                    if run_id == self.run_id:
                        self._finish_benchmark(int(code), Path(output), str(mode))
                elif kind == "error":
                    run_id, error = payload
                    if run_id == self.run_id:
                        self._set_idle()
                        messagebox.showerror("Testes OCR", str(error))
                elif kind == "preprocess_progress":
                    current, total, name = payload
                    self.active_stage_label.configure(text=f"Processando {current} de {total}: {name}")
                elif kind == "preprocess_folder_done":
                    total, destination = payload
                    self.folder_preprocess_button.configure(state="normal")
                    self.active_stage_label.configure(text=f"Lote concluído: {total} imagens processadas.")
                    messagebox.showinfo("Processar pasta", f"{total} imagens salvas em:\n{destination}")
                elif kind == "preprocess_folder_error":
                    self.folder_preprocess_button.configure(state="normal")
                    self.active_stage_label.configure(text="O processamento da pasta terminou com erro.")
                    messagebox.showerror("Processar pasta", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _set_idle(self) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.batch_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _finish_benchmark(self, code: int, output: Path, mode: str) -> None:
        self._set_idle()
        if code != 0 or not output.exists():
            self.bench_status.configure(text="O teste foi cancelado ou terminou com erro.")
            return
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
            model = data["models"][0]
            if mode == "batch":
                self._show_batch_results(data, model)
                elapsed = time.monotonic() - self.run_started_at
                self.bench_status.configure(
                    text=f"{data['image_count']} imagens concluídas em {elapsed:.1f} s. Clique em Certeza ou Tempo para ordenar."
                )
                return
            pred = data["results"][0]["predictions"][model]
            text = pred.get("text", pred.get("error", "Sem resultado"))
            self.result_text.configure(state="normal")
            self.result_text.insert("1.0", text or "Nenhum texto reconhecido")
            self.result_text.configure(state="disabled")
            if "error" in pred:
                self.result_meta.configure(text=f"{model} indisponível")
            else:
                confidence = pred.get("confidence")
                confidence_text = f"{confidence:.1%}" if confidence is not None else "não informada"
                self.result_meta.configure(text=f"Modelo: {model}\nConfiança: {confidence_text}\nTempo: {pred['latency_ms']:.0f} ms")
            elapsed = time.monotonic() - self.run_started_at
            self.bench_status.configure(text=f"Concluído em {elapsed:.1f} s.")
        except Exception as exc:
            messagebox.showerror("Resultados", f"Não foi possível ler o resultado: {exc}")

    def _show_batch_results(self, data: dict, model: str) -> None:
        self.batch_rows = []
        for entry in data["results"]:
            prediction = entry["predictions"][model]
            self.batch_rows.append({
                "image": entry["image"],
                "confidence": prediction.get("confidence"),
                "latency": prediction.get("latency_ms"),
                "text": prediction.get("text", prediction.get("error", "")),
                "model": model,
            })
        self._render_batch_rows()

    def _render_batch_rows(self) -> None:
        for item in self.batch_results.get_children():
            self.batch_results.delete(item)
        for index, row in enumerate(self.batch_rows):
            confidence = row["confidence"]
            latency = row["latency"]
            self.batch_results.insert("", "end", iid=str(index), values=(
                row["image"],
                f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "—",
                f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "—",
            ))

    def sort_batch_results(self, key: str) -> None:
        reverse = self.sort_reverse[key]
        available = [row for row in self.batch_rows if isinstance(row[key], (int, float))]
        unavailable = [row for row in self.batch_rows if not isinstance(row[key], (int, float))]
        available.sort(key=lambda row: float(row[key]), reverse=reverse)
        self.batch_rows = available + unavailable
        self.sort_reverse[key] = not reverse
        self._render_batch_rows()

    def open_batch_result(self, _event: tk.Event) -> None:
        selection = self.batch_results.selection()
        if not selection:
            return
        row = self.batch_rows[int(selection[0])]
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", str(row["text"]) or "Nenhum texto reconhecido")
        self.result_text.configure(state="disabled")
        confidence = row["confidence"]
        confidence_text = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "não informada"
        self.result_meta.configure(
            text=f"Imagem: {row['image']}\nModelo: {row['model']}\nConfiança: {confidence_text}"
        )


def main() -> None:
    app = OcrDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
