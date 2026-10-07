from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import PipelineConfig
from src.alignment import AlignmentError, align_to_template
from src.document_types import DocumentType, collect_dataset_pdfs, discover_document_types
from src.pdf import keep_first_page_in_place, pdf_to_image, standardize_size
from src.pipeline import run_pipeline
from src.ocr_bench import BACKEND_FACTORIES
from src.preprocess import preprocess_for_ocr


class PipelineRunnerFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.document_types = discover_document_types()
        self.document_type_var = tk.StringVar()
        self.mapping_choice_var = tk.StringVar()
        self.document_var = tk.StringVar()
        self.template_var = tk.StringVar()
        self.field_map_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output_gui"))
        self.document_id_var = tk.StringVar(value="document_001")
        self.preset_var = tk.StringVar()
        self.use_preset_var = tk.BooleanVar(value=False)
        self.hue_min_var = tk.DoubleVar(value=80)
        self.hue_max_var = tk.DoubleVar(value=140)
        self.sat_min_var = tk.DoubleVar(value=10)
        self.gray_max_var = tk.DoubleVar(value=220)
        self.ink_threshold_var = tk.DoubleVar(value=250)
        self.noise_area_var = tk.DoubleVar(value=4)
        self.close_kernel_var = tk.DoubleVar(value=3)
        self.background_threshold_var = tk.DoubleVar(value=245)
        self.brightness_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=0)
        self.exposure_var = tk.DoubleVar(value=0)
        self.grayscale_only_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Selecione os arquivos e execute o pipeline.")
        self.viewer_info_var = tk.StringVar(value="Nenhum PDF carregado")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.viewer_image: np.ndarray | None = None
        self.viewer_source_image: np.ndarray | None = None
        self.viewer_dpi = 120
        self.viewer_photo: ImageTk.PhotoImage | None = None
        self.viewer_scale = 1.0
        self.viewer_offset = (0, 0)
        self._viewer_cache_key: tuple[int, int, int] | None = None
        self.manifest: dict | None = None
        self.output_path: Path | None = None
        self.hovered_index: int | None = None
        self.pinned_index: int | None = None
        self.crop_photo: ImageTk.PhotoImage | None = None
        self.processed_crop_photo: ImageTk.PhotoImage | None = None
        self.preset_preview_job: str | None = None
        self.preset_dirty = False
        self.document_paths: list[Path] = []
        self.document_index = 0
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, style="Panel.TFrame", padding=(10, 8))
        top.pack(fill="x", pady=(10, 8))
        self.left_toggle = ttk.Button(top, text="Ocultar opções ◀", command=self._toggle_options)
        self.left_toggle.pack(side="left")
        self.right_toggle = ttk.Button(top, text="Ocultar detalhes ▶", command=self._toggle_details)
        self.right_toggle.pack(side="right")
        ttk.Label(top, textvariable=self.status_var, style="Panel.TLabel", anchor="center").pack(
            side="left", fill="x", expand=True, padx=10
        )

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.options_panel = ttk.Frame(body, style="Panel.TFrame", width=340)
        self.options_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        self.options_panel.grid_propagate(False)
        inputs = self._scrollable_column(self.options_panel)
        inputs.columnconfigure(0, weight=1)
        ttk.Label(inputs, text="Tipo de documento", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        type_values = tuple(item.label for item in self.document_types.values())
        self.document_type_box = ttk.Combobox(
            inputs, textvariable=self.document_type_var, values=type_values, state="readonly",
        )
        self.document_type_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.document_type_box.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_document_type())
        ttk.Label(inputs, text="Documentos para OCR", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        self.document_button = ttk.Button(
            inputs, text="Selecionar documento para aplicar o OCR", style="Accent.TButton",
            command=self._browse_document,
        )
        self.document_button.grid(row=3, column=0, sticky="ew", pady=3)
        ttk.Button(inputs, text="Carregar pasta de documentos…", command=self._browse_document_folder).grid(
            row=4, column=0, sticky="ew", pady=3
        )
        self.document_label = ttk.Label(inputs, text="Nenhum documento preenchido", style="Panel.TLabel", wraplength=300)
        self.document_label.grid(row=5, column=0, sticky="ew", pady=(3, 3))
        navigation = ttk.Frame(inputs, style="Panel.TFrame")
        navigation.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        self.previous_document_button = ttk.Button(
            navigation, text="←", width=4, state="disabled", command=lambda: self._navigate_document(-1),
        )
        self.previous_document_button.pack(side="left")
        self.document_counter = ttk.Label(navigation, text="", style="Panel.TLabel", anchor="center")
        self.document_counter.pack(side="left", fill="x", expand=True)
        self.next_document_button = ttk.Button(
            navigation, text="→", width=4, state="disabled", command=lambda: self._navigate_document(1),
        )
        self.next_document_button.pack(side="right")

        ttk.Separator(inputs).grid(row=7, column=0, sticky="ew", pady=5)
        ttk.Label(inputs, text="Mapeamento e execução", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold")).grid(
            row=8, column=0, sticky="w", pady=(5, 4)
        )
        self.mapping_box = ttk.Combobox(inputs, textvariable=self.mapping_choice_var, state="readonly")
        self.mapping_box.grid(row=9, column=0, sticky="ew", pady=3)
        self.mapping_box.bind("<<ComboboxSelected>>", lambda _event: self._select_mapping())
        mapping_actions = ttk.Frame(inputs, style="Panel.TFrame")
        mapping_actions.grid(row=10, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(mapping_actions, text="Importar mapeamento…", command=self._import_mapping).pack(side="left")
        ttk.Button(mapping_actions, text="Abrir outro…", command=self._browse_field_map).pack(side="right")
        self._compact_path_row(inputs, 11, "Pasta de saída", self.output_var, self._browse_output)

        options = ttk.Frame(inputs, style="Panel.TFrame")
        options.grid(row=12, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(options, text="ID do documento", style="Panel.TLabel").pack(side="left")
        ttk.Entry(options, textvariable=self.document_id_var, width=15).pack(side="left", padx=6)
        self.run_button = ttk.Button(options, text="Executar pipeline", style="Accent.TButton", command=self.run)
        self.run_button.pack(side="right")
        self.progress = ttk.Progressbar(inputs, mode="indeterminate")
        self.progress.grid(row=13, column=0, sticky="ew", pady=(6, 0))

        selectors = ttk.Frame(inputs, style="Panel.TFrame")
        selectors.grid(row=14, column=0, sticky="ew", pady=(10, 0))
        self.model_list = self._selection_list(selectors, "Modelos OCR", tuple(BACKEND_FACTORIES), 20)
        self.ocr_type_list = self._selection_list(selectors, "OCR nos tipos", ("Todos", "text"), 20)
        self.preset_type_list = self._selection_list(selectors, "Preset nos tipos", ("Todos", "text"), 20)
        self.model_list.selection_set(0)
        self.ocr_type_list.selection_set(0)
        self.preset_type_list.selection_set(0)
        self.preset_type_list.bind("<<ListboxSelect>>", lambda _e: self._schedule_preset_preview())
        self._build_preset_editor(inputs, 15)

        viewer = ttk.Frame(body)
        viewer.grid(row=0, column=1, sticky="nsew")
        self.details_panel = ttk.Frame(body, style="Panel.TFrame", width=400)
        self.details_panel.grid(row=0, column=2, sticky="nse", padx=(8, 0))
        self.details_panel.grid_propagate(False)
        details = self._scrollable_column(self.details_panel)

        viewer_header = ttk.Frame(viewer, style="Panel.TFrame", padding=(10, 6))
        viewer_header.pack(fill="x", pady=(0, 4))
        ttk.Label(
            viewer_header, textvariable=self.viewer_info_var, style="Panel.TLabel", anchor="center",
        ).pack(fill="x")
        self.canvas = tk.Canvas(viewer, bg="#0e1012", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_viewer())
        self.canvas.bind("<Motion>", self._hover_field)
        self.canvas.bind("<Button-1>", self._pin_field)
        self.canvas.bind("<Leave>", lambda _e: self._clear_hover())
        self.empty_state = ttk.Frame(viewer, style="Panel.TFrame", padding=24)
        ttk.Label(
            self.empty_state, text="Escolha o tipo e o documento preenchido",
            style="Panel.TLabel", font=("TkDefaultFont", 13, "bold"),
        ).pack(pady=(0, 12))
        ttk.Button(
            self.empty_state, text="Selecionar documento para aplicar o OCR",
            style="Accent.TButton", command=self._browse_document,
        ).pack(fill="x", pady=4)
        ttk.Button(
            self.empty_state, text="Usar tipo de documento selecionado",
            command=self._load_selected_document_type,
        ).pack(fill="x", pady=4)
        self.empty_state.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(details, text="Campo sob o cursor", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=14, pady=(14, 8))
        self.field_title = ttk.Label(details, text="Passe o mouse ou clique para fixar", style="Panel.TLabel", wraplength=355)
        self.field_title.pack(fill="x", padx=14)
        crop_comparison = ttk.Frame(details, style="Panel.TFrame")
        crop_comparison.pack(fill="x", padx=14, pady=10)
        original_box = ttk.Frame(crop_comparison, style="Panel.TFrame")
        processed_box = ttk.Frame(crop_comparison, style="Panel.TFrame")
        original_box.pack(fill="x", expand=True, pady=(0, 8))
        processed_box.pack(fill="x", expand=True)
        ttk.Label(original_box, text="Original", style="Panel.TLabel").pack(anchor="w")
        ttk.Label(processed_box, text="Prévia atual do preset", style="Panel.TLabel").pack(anchor="w")
        self.crop_label = ttk.Label(original_box, text="", style="Panel.TLabel", anchor="center")
        self.crop_label.pack(fill="x")
        self.processed_crop_label = ttk.Label(processed_box, text="", style="Panel.TLabel", anchor="center")
        self.processed_crop_label.pack(fill="x")
        ttk.Label(details, text="Resultados OCR por modelo", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=14)
        self.ocr_text = tk.Text(details, height=10, wrap="word", bg="#292d33", fg="#f2f3f5", relief="flat", padx=8, pady=8, state="disabled")
        self.ocr_text.pack(fill="x", padx=14, pady=(5, 8))
        self.field_meta = ttk.Label(details, text="", style="Panel.TLabel", wraplength=355)
        self.field_meta.pack(fill="x", padx=14)
        ttk.Label(
            details,
            text="A visualização usa o documento preenchido alinhado ao template; o conteúdo continua sendo o original, mas as caixas ficam nas coordenadas corretas do mapa.",
            style="Panel.TLabel", wraplength=355,
        ).pack(fill="x", padx=14, side="bottom", pady=14)

        self.after_idle(self._render_viewer)
        if self.document_types:
            first = next(iter(self.document_types.values()))
            self.document_type_var.set(first.label)
            self._load_selected_document_type()

    @staticmethod
    def _scrollable_column(parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(parent, bg="#202328", highlightthickness=0, width=320)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, style="Panel.TFrame", padding=12)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        def wheel(event: tk.Event) -> str | None:
            x, y = event.x_root, event.y_root
            left, top = parent.winfo_rootx(), parent.winfo_rooty()
            if not (left <= x < left + parent.winfo_width() and top <= y < top + parent.winfo_height()):
                return None
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
            canvas.yview_scroll(direction, "units")
            return "break"
        canvas.bind_all("<MouseWheel>", wheel, add="+")
        canvas.bind_all("<Button-4>", wheel, add="+")
        canvas.bind_all("<Button-5>", wheel, add="+")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return content

    def _toggle_options(self) -> None:
        if self.options_panel.winfo_ismapped():
            self.options_panel.grid_remove()
            self.left_toggle.configure(text="Mostrar opções ▶")
        else:
            self.options_panel.grid()
            self.left_toggle.configure(text="Ocultar opções ◀")

    def _toggle_details(self) -> None:
        if self.details_panel.winfo_ismapped():
            self.details_panel.grid_remove()
            self.right_toggle.configure(text="Mostrar detalhes ◀")
        else:
            self.details_panel.grid()
            self.right_toggle.configure(text="Ocultar detalhes ▶")

    def _compact_path_row(self, parent: ttk.Frame, row: int, label: str,
                          variable: tk.StringVar, command) -> None:
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Button(box, text=label + "…", command=command).pack(side="left")
        ttk.Label(box, textvariable=variable, style="Panel.TLabel", anchor="w").pack(
            side="left", fill="x", expand=True, padx=5
        )

    def _selection_list(self, parent: ttk.Frame, label: str, values: tuple[str, ...], width: int) -> tk.Listbox:
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.pack(fill="x", pady=(0, 7))
        ttk.Label(box, text=label + " (Ctrl+clique)", style="Panel.TLabel").pack(anchor="w")
        list_frame = ttk.Frame(box, style="Panel.TFrame")
        list_frame.pack()
        widget = tk.Listbox(
            list_frame, selectmode="extended", exportselection=False, height=3, width=width,
            bg="#292d33", fg="#f2f3f5", selectbackground="#34446f", relief="flat",
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scrollbar.set)
        for value in values:
            widget.insert("end", value)
        widget.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="left", fill="y")
        return widget

    def _build_preset_editor(self, parent: ttk.Frame, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="Preset de pré-processamento", padding=(8, 5))
        panel.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            panel, text="Aplicar preset ao OCR e à prévia", variable=self.use_preset_var,
            command=self._schedule_preset_preview,
        ).pack(anchor="w", pady=(0, 5))
        primary_actions = ttk.Frame(panel, style="Panel.TFrame")
        primary_actions.pack(fill="x", pady=(0, 5))
        ttk.Button(
            primary_actions, text="Aplicar ao PDF", style="Accent.TButton",
            command=self._apply_document_preprocess,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        ttk.Button(
            primary_actions, text="Desfazer", command=self._undo_document_preprocess,
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        ttk.Button(
            primary_actions, text="Resetar preset", command=self._reset_preset_defaults,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.save_preset_button = ttk.Button(
            primary_actions, text="Salvar preset…", command=self._save_preset, state="disabled",
        )
        self.save_preset_button.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=2)
        primary_actions.columnconfigure(0, weight=1)
        primary_actions.columnconfigure(1, weight=1)
        ttk.Button(
            panel, text="Editar parâmetros ▾", command=self._toggle_preset_editor,
        ).pack(fill="x")
        self.preset_editor_body = ttk.Frame(panel, style="Panel.TFrame")
        actions = ttk.Frame(self.preset_editor_body, style="Panel.TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Carregar preset…", command=self._browse_preset).pack(fill="x", pady=2)
        self.grayscale_button = ttk.Button(actions, text="Apenas cinza", command=self._toggle_grayscale_only)
        self.grayscale_button.pack(fill="x", pady=2)
        self.preset_name_label = ttk.Label(actions, text="Valores padrão", style="Panel.TLabel")
        self.preset_name_label.pack(fill="x", pady=3)

        sliders = ttk.Frame(self.preset_editor_body, style="Panel.TFrame")
        sliders.pack(fill="x", pady=(5, 0))

        def add_slider(grid_row: int, column: int, label: str, variable: tk.DoubleVar,
                       start: float, end: float, decimals: int = 0) -> None:
            box = ttk.Frame(sliders, style="Panel.TFrame")
            box.grid(row=grid_row, column=column, sticky="ew", padx=5, pady=2)
            ttk.Label(box, text=label, style="Panel.TLabel", width=18).pack(side="left")
            formatter = lambda value: f"{value:.{decimals}f}"
            value_label = ttk.Label(box, text=formatter(variable.get()), style="Panel.TLabel", width=5, anchor="e")
            value_label.pack(side="right")
            variable.trace_add(
                "write",
                lambda *_args, shown=value_label, current=variable, digits=decimals:
                    self._update_slider_label(shown, current, digits),
            )
            scale = ttk.Scale(box, variable=variable, from_=start, to=end)
            scale.pack(side="right", fill="x", expand=True, padx=5)
            scale.bind("<ButtonRelease-1>", lambda _event: self._preset_edited())
            scale.bind("<KeyRelease>", lambda _event: self._preset_edited())

        controls = (
            ("Azul mínimo", self.hue_min_var, 0, 179), ("Azul máximo", self.hue_max_var, 0, 179),
            ("Saturação", self.sat_min_var, 0, 255), ("Escuro", self.gray_max_var, 0, 255),
            ("Tolerância da tinta", self.ink_threshold_var, 0, 255),
            ("Área mínima", self.noise_area_var, 1, 500),
            ("Fechamento", self.close_kernel_var, 1, 15),
            ("Tolerância do fundo", self.background_threshold_var, 0, 255),
            ("Brilho", self.brightness_var, -100, 100),
            ("Contraste (%)", self.contrast_var, -100, 100),
            ("Exposição (EV)", self.exposure_var, -3, 3, 1),
        )
        for index, control in enumerate(controls):
            label, variable, start, end, *decimals = control
            add_slider(index, 0, label, variable, start, end, decimals[0] if decimals else 0)
        sliders.columnconfigure(0, weight=1)

    def _toggle_preset_editor(self) -> None:
        if self.preset_editor_body.winfo_ismapped():
            self.preset_editor_body.pack_forget()
        else:
            self.preset_editor_body.pack(fill="x", pady=(6, 0))

    def _preset_parameters(self) -> dict:
        return {
            "hue_range": [round(self.hue_min_var.get()), round(self.hue_max_var.get())],
            "sat_min": round(self.sat_min_var.get()), "gray_max": round(self.gray_max_var.get()),
            "ink_threshold": round(self.ink_threshold_var.get()),
            "min_component_area": round(self.noise_area_var.get()),
            "close_kernel": round(self.close_kernel_var.get()),
            "background_threshold": round(self.background_threshold_var.get()),
            "brightness": round(self.brightness_var.get()),
            "contrast": round(self.contrast_var.get()),
            "exposure": round(self.exposure_var.get(), 1),
            "grayscale_only": self.grayscale_only_var.get(),
        }

    @staticmethod
    def _update_slider_label(value_label: ttk.Label, variable: tk.DoubleVar,
                             decimals: int = 0) -> None:
        value_label.configure(text=f"{variable.get():.{decimals}f}")

    def _toggle_grayscale_only(self) -> None:
        enabled = not self.grayscale_only_var.get()
        self.grayscale_only_var.set(enabled)
        self.grayscale_button.configure(text="Cinza ativado ✓" if enabled else "Apenas cinza")
        self._preset_edited()

    def _preset_edited(self) -> None:
        self.preset_dirty = True
        self.use_preset_var.set(True)
        self.save_preset_button.configure(state="normal")
        self.preset_name_label.configure(text="Preset editado · ainda não salvo")
        self._schedule_preset_preview()

    def _reset_preset_defaults(self) -> None:
        if self.preset_preview_job is not None:
            self.after_cancel(self.preset_preview_job)
            self.preset_preview_job = None
        defaults = (
            (self.hue_min_var, 80), (self.hue_max_var, 140),
            (self.sat_min_var, 10), (self.gray_max_var, 220),
            (self.ink_threshold_var, 250), (self.noise_area_var, 4),
            (self.close_kernel_var, 3), (self.background_threshold_var, 245),
            (self.brightness_var, 0), (self.contrast_var, 0), (self.exposure_var, 0),
        )
        for variable, value in defaults:
            variable.set(value)
        self.grayscale_only_var.set(False)
        self.use_preset_var.set(False)
        self.preset_dirty = False
        self.save_preset_button.configure(state="disabled")
        self.grayscale_button.configure(text="Apenas cinza")
        self.preset_var.set("")
        self.preset_name_label.configure(text="Valores padrão")
        self.processed_crop_photo = None
        self.processed_crop_label.configure(image="", text="Preset desativado", foreground="#a9afb8")
        if self.viewer_source_image is not None:
            self.viewer_image = self.viewer_source_image.copy()
            self._viewer_cache_key = None
            self._render_viewer()
        if not self.preset_editor_body.winfo_ismapped():
            self.preset_editor_body.pack(fill="x", pady=(6, 0))
        self.status_var.set("Preset resetado. Ajuste os parâmetros para começar novamente.")

    def _schedule_preset_preview(self) -> None:
        if self.preset_preview_job is not None:
            self.after_cancel(self.preset_preview_job)
        self.preset_preview_job = self.after(100, self._update_preset_preview)

    def _preset_applies_to_field(self, field: dict) -> bool:
        if not self.use_preset_var.get():
            return False
        selected_types = self._selected_values(self.preset_type_list)
        return "Todos" in selected_types or field.get("data_type", "text") in selected_types

    def _update_preset_preview(self) -> None:
        self.preset_preview_job = None
        self._refresh_document_preprocess()
        if self.hovered_index is None or not self.manifest or not self.output_path:
            return
        fields = self.manifest.get("fields", [])
        if self.hovered_index >= len(fields):
            return
        field = fields[self.hovered_index]
        crop_path = field.get("crop_path")
        original = cv2.imread(str(self.output_path / crop_path), cv2.IMREAD_COLOR) if crop_path else None
        if original is None:
            return
        if not self._preset_applies_to_field(field):
            self.processed_crop_photo = None
            reason = "Preset desativado" if not self.use_preset_var.get() else "Não aplicado a este tipo"
            self.processed_crop_label.configure(image="", text=reason, foreground="#a9afb8")
            return
        try:
            processed = preprocess_for_ocr(
                original, mode="traditional", **self._preset_parameters()
            )
            self.processed_crop_photo = self._crop_photo(processed)
            self.processed_crop_label.configure(image=self.processed_crop_photo, text="")
        except Exception as exc:
            self.processed_crop_photo = None
            self.processed_crop_label.configure(image="", text=f"Erro: {exc}")

    def _apply_document_preprocess(self) -> None:
        if self.viewer_source_image is None:
            messagebox.showinfo("Pré-processamento", "Abra um documento primeiro.")
            return
        self.use_preset_var.set(True)
        self._refresh_document_preprocess()

    def _undo_document_preprocess(self) -> None:
        if self.viewer_source_image is None:
            return
        self.use_preset_var.set(False)
        self.viewer_image = self.viewer_source_image.copy()
        self._viewer_cache_key = None
        self._render_viewer()
        self.status_var.set("Pré-processamento desfeito; exibindo o documento original.")

    def _refresh_document_preprocess(self) -> None:
        if self.viewer_source_image is None:
            return
        if not self.use_preset_var.get():
            self.viewer_image = self.viewer_source_image.copy()
        else:
            try:
                processed = preprocess_for_ocr(
                    self.viewer_source_image, mode="traditional", margin=0,
                    **self._preset_parameters(),
                )
                self.viewer_image = processed
            except Exception as exc:
                messagebox.showerror("Pré-processamento", str(exc))
                return
        self._viewer_cache_key = None
        self._render_viewer()


    def _browse_document(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            document = Path(path)
            if not self._prepare_document_pdf(document):
                return
            self.document_paths = [document]
            self.document_index = 0
            self._show_current_document()

    def _browse_document_folder(self) -> None:
        current_type = self._current_document_type()
        initial = str(current_type.dataset) if current_type and current_type.dataset.is_dir() else None
        path = filedialog.askdirectory(initialdir=initial)
        if not path:
            return
        documents = collect_dataset_pdfs(Path(path))
        if not documents:
            messagebox.showinfo("Pasta de documentos", "A pasta selecionada não contém PDFs.")
            return
        prepared = [document for document in documents if self._prepare_document_pdf(document)]
        if not prepared:
            messagebox.showerror("Pasta de documentos", "Nenhum PDF válido pôde ser preparado.")
            return
        self.document_paths = prepared
        self.document_index = 0
        self._show_current_document()

    def _prepare_document_pdf(self, document: Path) -> bool:
        try:
            page_count, changed = keep_first_page_in_place(str(document))
            if changed:
                self.status_var.set(
                    f"{document.name}: mantida a primeira página; {page_count - 1} página(s) removida(s)."
                )
            return True
        except Exception as exc:
            messagebox.showerror("Preparação do PDF", f"Não foi possível preparar {document.name}: {exc}")
            return False

    def _navigate_document(self, step: int) -> None:
        if not self.document_paths:
            return
        self.document_index = (self.document_index + step) % len(self.document_paths)
        self._show_current_document()

    def _show_current_document(self) -> None:
        document = self.document_paths[self.document_index]
        self.document_var.set(str(document))
        self.document_label.configure(text=document.name)
        total = len(self.document_paths)
        self.document_counter.configure(text=f"{self.document_index + 1} de {total}" if total > 1 else "1 documento")
        state = "normal" if total > 1 else "disabled"
        self.previous_document_button.configure(state=state)
        self.next_document_button.configure(state=state)
        self.document_id_var.set(document.stem)
        self._preview_selected_document()

    def _current_document_type(self) -> DocumentType | None:
        selected = self.document_type_var.get()
        return next((item for item in self.document_types.values() if item.label == selected), None)

    def _load_selected_document_type(self) -> None:
        document_type = self._current_document_type()
        if document_type is None:
            if not self.document_types:
                messagebox.showerror("Tipos de documento", "Nenhum tipo com template.pdf foi encontrado em documents/.")
            return
        self.template_var.set(str(document_type.template))
        mapping_names = tuple(path.name for path in document_type.mappings)
        self.mapping_box.configure(values=mapping_names)
        if mapping_names:
            current_name = Path(self.field_map_var.get()).name
            selected = current_name if current_name in mapping_names else mapping_names[0]
            self.mapping_choice_var.set(selected)
            self._select_mapping()
        else:
            self.mapping_choice_var.set("")
            self.field_map_var.set("")
            self.manifest = None
        self._preview_selected_document()

    def _select_mapping(self) -> None:
        document_type = self._current_document_type()
        if document_type is None:
            return
        selected = document_type.root / "mappings" / self.mapping_choice_var.get()
        if selected.is_file():
            self.field_map_var.set(str(selected))
            self._refresh_data_types(selected)
            self._load_field_map_preview(selected)

    def _import_mapping(self) -> None:
        document_type = self._current_document_type()
        if document_type is None:
            messagebox.showinfo("Mapeamento", "Selecione primeiro um tipo de documento.")
            return
        path = filedialog.askopenfilename(filetypes=[("Mapa JSON", "*.json")])
        if not path:
            return
        source = Path(path)
        destination = document_type.root / "mappings" / source.name
        if destination.exists() and source.resolve() != destination.resolve():
            if not messagebox.askyesno("Mapeamento", f"Substituir {destination.name}?"):
                return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        self.document_types = discover_document_types()
        refreshed = self._current_document_type()
        if refreshed is not None:
            self.mapping_box.configure(values=tuple(item.name for item in refreshed.mappings))
        self.mapping_choice_var.set(destination.name)
        self._select_mapping()

    def _browse_template(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if path:
            self.template_var.set(path)
            self.template_label.configure(text=Path(path).name)
            self._preview_selected_document()

    def _browse_field_map(self) -> None:
        document_type = self._current_document_type()
        initial = str(document_type.root / "mappings") if document_type else None
        path = filedialog.askopenfilename(initialdir=initial, filetypes=[("Mapa JSON", "*.json")])
        if path:
            self.field_map_var.set(path)
            self._refresh_data_types(Path(path))
            self._load_field_map_preview(Path(path))

    def _preview_selected_document(self) -> None:
        document = Path(self.document_var.get()).expanduser()
        template = Path(self.template_var.get()).expanduser()
        if not document.is_file() or not template.is_file():
            self._render_viewer()
            return
        try:
            self.status_var.set("Abrindo e alinhando documento ao template…")
            self.viewer_dpi = 120
            document_page = pdf_to_image(str(document), dpi=self.viewer_dpi)
            template_page = pdf_to_image(str(template), dpi=self.viewer_dpi)
            document_image, template_image = standardize_size(
                document_page.image, template_page.image,
            )
            preview_config = PipelineConfig(dpi=self.viewer_dpi)
            aligned = align_to_template(template_image, document_image, preview_config)
            self.viewer_source_image = aligned.warped_image
            self.viewer_image = self.viewer_source_image.copy()
            self._update_viewer_info(
                document, document_page.image, self.viewer_source_image,
                match_count=aligned.match_count, inlier_ratio=aligned.inlier_ratio,
            )
            self._viewer_cache_key = None
            self.empty_state.place_forget()
            self._render_viewer()
            self.status_var.set("Documento alinhado ao template; o mapeamento usa as coordenadas normalizadas.")
        except AlignmentError as exc:
            messagebox.showerror("Alinhamento", f"Não foi possível alinhar {document.name}: {exc}")
        except Exception as exc:
            messagebox.showerror("Documento", f"Não foi possível abrir o PDF: {exc}")

    def _update_viewer_info(self, document: Path, original: np.ndarray,
                            normalized: np.ndarray | None = None,
                            match_count: int | None = None,
                            inlier_ratio: float | None = None) -> None:
        original_height, original_width = original.shape[:2]
        size_mb = document.stat().st_size / (1024 * 1024)
        parts = [
            document.name, f"original {original_width} × {original_height} px",
            f"{size_mb:.2f} MB", "1 página",
        ]
        if normalized is not None:
            height, width = normalized.shape[:2]
            parts.append(f"normalizado {width} × {height} px")
        if match_count is not None and inlier_ratio is not None:
            parts.append(f"alinhamento {match_count} pontos · {inlier_ratio:.0%}")
        self.viewer_info_var.set("  ·  ".join(parts))

    def _load_field_map_preview(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            source_dpi = data.get("dpi", 300)
            ratio = self.viewer_dpi / source_dpi
            fields = []
            for field in data.get("fields", []):
                item = dict(field)
                item["bbox"] = [round(value * ratio) for value in field["bbox"]]
                item["elements"] = [
                    {
                        **element,
                        "bbox": [round(value * ratio) for value in element["bbox"]],
                    }
                    for element in field.get("elements", [])
                ]
                item.setdefault("filled", False)
                fields.append(item)
            self.manifest = {"fields": fields}
            self.hovered_index = None
            self.pinned_index = None
            self._viewer_cache_key = None
            self._render_viewer()
        except Exception as exc:
            messagebox.showerror("Mapa de campos", f"Não foi possível exibir o mapa: {exc}")

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def _browse_preset(self) -> None:
        document_type = self._current_document_type()
        initial = str(document_type.root / "presets") if document_type else None
        path = filedialog.askopenfilename(initialdir=initial, filetypes=[("Preset JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("preset_version") != 1 or data.get("mode") != "traditional":
                raise ValueError("Formato de preset incompatível")
            values = data["parameters"]
            assignments = (
                (self.hue_min_var, values["hue_range"][0]),
                (self.hue_max_var, values["hue_range"][1]),
                (self.sat_min_var, values["sat_min"]),
                (self.gray_max_var, values["gray_max"]),
                (self.ink_threshold_var, values["ink_threshold"]),
                (self.noise_area_var, values["min_component_area"]),
                (self.close_kernel_var, values["close_kernel"]),
                (self.background_threshold_var, values["background_threshold"]),
                (self.brightness_var, values.get("brightness", 0)),
                (self.contrast_var, values.get("contrast", 0)),
                (self.exposure_var, values.get("exposure", 0)),
            )
            for variable, value in assignments:
                variable.set(int(value))
            self.preset_var.set(path)
            grayscale_only = bool(values.get("grayscale_only", False))
            self.grayscale_only_var.set(grayscale_only)
            self.grayscale_button.configure(text="Cinza ativado ✓" if grayscale_only else "Apenas cinza")
            self.preset_name_label.configure(text=Path(path).name)
            self.use_preset_var.set(True)
            self.preset_dirty = False
            self.save_preset_button.configure(state="disabled")
            self._schedule_preset_preview()
        except Exception as exc:
            messagebox.showerror("Preset", f"Não foi possível carregar o preset: {exc}")

    def _save_preset(self) -> None:
        document_type = self._current_document_type()
        initial = str(document_type.root / "presets") if document_type else None
        path = filedialog.asksaveasfilename(
            initialdir=initial, defaultextension=".json", initialfile="preset_preprocessamento.json",
            filetypes=[("Preset JSON", "*.json")],
        )
        if not path:
            return
        try:
            data = {"preset_version": 1, "mode": "traditional", "parameters": self._preset_parameters()}
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.preset_var.set(path)
            self.preset_name_label.configure(text=Path(path).name)
            self.use_preset_var.set(True)
            self.preset_dirty = False
            self.save_preset_button.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Preset", f"Não foi possível salvar o preset: {exc}")

    def _refresh_data_types(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            types = sorted({field.get("data_type", "text") for field in data.get("fields", [])})
            values = ["Todos", *types]
            for widget in (self.ocr_type_list, self.preset_type_list):
                widget.delete(0, "end")
                for value in values:
                    widget.insert("end", value)
                widget.selection_set(0)
        except Exception as exc:
            messagebox.showerror("Mapa de campos", f"Não foi possível ler os tipos: {exc}")

    @staticmethod
    def _selected_values(widget: tk.Listbox) -> list[str]:
        return [str(widget.get(index)) for index in widget.curselection()]

    def run(self) -> None:
        document = Path(self.document_var.get()).expanduser()
        template = Path(self.template_var.get()).expanduser()
        field_map = Path(self.field_map_var.get()).expanduser()
        output = Path(self.output_var.get()).expanduser()
        document_id = self.document_id_var.get().strip()
        if not document.is_file() or not template.is_file() or not field_map.is_file():
            messagebox.showerror("Pipeline", "Selecione o documento preenchido, o vazio e o mapa JSON.")
            return
        if not document_id:
            messagebox.showerror("Pipeline", "Informe o ID do documento.")
            return
        models = self._selected_values(self.model_list)
        ocr_types = self._selected_values(self.ocr_type_list)
        preset_types = self._selected_values(self.preset_type_list)
        if not models or not ocr_types:
            messagebox.showerror("Pipeline", "Selecione ao menos um modelo e um tipo para OCR.")
            return
        preset_parameters = self._preset_parameters() if self.use_preset_var.get() else None
        preset_source = self.preset_var.get().strip() or "editado na interface"
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Executando alinhamento, detecção e OCR…")
        threading.Thread(
            target=self._worker,
            args=(document, template, field_map, output, document_id, models, ocr_types,
                  preset_parameters, preset_source, preset_types),
            daemon=True,
        ).start()

    def _worker(self, document: Path, template: Path, field_map: Path,
                output: Path, document_id: str, models: list[str], ocr_types: list[str],
                preset_parameters: dict | None, preset_source: str,
                preset_types: list[str]) -> None:
        try:
            run_pipeline(
                str(document), str(template), str(output), PipelineConfig(), document_id,
                field_map_path=str(field_map), skip_ocr=True,
            )
            manifest_path = output / f"{document_id}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            backends = {}
            for model in models:
                self.events.put(("status", f"Carregando modelo {model}…"))
                try:
                    backends[model] = BACKEND_FACTORIES[model]()
                except Exception as exc:
                    backends[model] = exc

            selected_all = "Todos" in ocr_types
            preset_all = "Todos" in preset_types
            aligned_image = cv2.imread(str(output / "debug" / "aligned.png"), cv2.IMREAD_COLOR)
            if aligned_image is None:
                raise OSError("Não foi possível abrir a imagem alinhada para o OCR dos campos")
            selected_fields = [
                field for field in manifest.get("fields", [])
                if field.get("filled") and (selected_all or field.get("data_type", "text") in ocr_types)
            ]
            for field_index, field in enumerate(selected_fields, start=1):
                data_type = field.get("data_type", "text")
                if data_type in {"digit_sequence", "text_sequence"} and field.get("elements"):
                    predictions = self._recognize_sequence_elements(
                        aligned_image, field, backends, models,
                        preset_parameters if preset_parameters is not None and (
                            preset_all or data_type in preset_types
                        ) else None,
                        output,
                    )
                    field["ocr_predictions"] = predictions
                    first = predictions.get(models[0], {})
                    field["ocr_text"] = first.get("text", "")
                    field["confidence"] = first.get("confidence") or 0.0
                    self.events.put(("progress", (field_index, len(selected_fields), field.get("label", ""))))
                    continue
                crop = cv2.imread(str(output / field["crop_path"]), cv2.IMREAD_COLOR)
                if crop is None:
                    continue
                if preset_parameters is not None and (preset_all or data_type in preset_types):
                    crop = preprocess_for_ocr(crop, mode="traditional", **preset_parameters)
                    processed_dir = output / "crops_preprocessed"
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    processed_path = processed_dir / Path(field["crop_path"]).name
                    if not cv2.imwrite(str(processed_path), crop):
                        raise OSError(f"Não foi possível salvar {processed_path}")
                    field["preprocessed_crop_path"] = str(
                        Path("crops_preprocessed") / processed_path.name
                    )
                if crop.ndim == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                predictions = {}
                for model, backend in backends.items():
                    if isinstance(backend, Exception):
                        predictions[model] = {"error": str(backend), "text": "", "confidence": None, "latency_ms": None}
                        continue
                    started = time.perf_counter()
                    try:
                        prediction = backend.recognize(crop)
                        predictions[model] = {
                            "text": prediction.text, "confidence": prediction.confidence,
                            "latency_ms": prediction.latency_ms,
                        }
                    except Exception as exc:
                        predictions[model] = {
                            "error": str(exc), "text": "", "confidence": None,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                field["ocr_predictions"] = predictions
                first = predictions.get(models[0], {})
                field["ocr_text"] = first.get("text", "")
                field["confidence"] = first.get("confidence") or 0.0
                self.events.put(("progress", (field_index, len(selected_fields), field.get("label", ""))))
            manifest["ocr_run"] = {
                "models": models, "data_types": ocr_types,
                "preset": preset_source if preset_parameters is not None else None,
                "preset_parameters": preset_parameters,
                "preset_data_types": preset_types,
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.events.put(("done", (manifest_path, output)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    @staticmethod
    def _recognize_sequence_elements(aligned_image: np.ndarray, field: dict,
                                     backends: dict, models: list[str],
                                     preset_parameters: dict | None,
                                     output: Path) -> dict[str, dict]:
        height, width = aligned_image.shape[:2]
        elements = sorted(
            field.get("elements", []),
            key=lambda item: (item.get("index", 10_000), item["bbox"][0]),
        )
        per_model: dict[str, list[dict]] = {model: [] for model in models}
        processed_dir = output / "crops_preprocessed" / "elements"
        for element_position, element in enumerate(elements):
            x1, y1, x2, y2 = element["bbox"]
            padding = 4
            crop = aligned_image[
                max(0, y1 - padding):min(height, y2 + padding),
                max(0, x1 - padding):min(width, x2 + padding),
            ].copy()
            if crop.size == 0:
                raise ValueError(f"Subelemento sem área válida: {element.get('element_id')}")
            if preset_parameters is not None:
                crop = preprocess_for_ocr(crop, mode="traditional", **preset_parameters)
                processed_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(processed_dir / f"{field.get('field_id')}_{element_position:03d}.png"), crop,
                )
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            element_predictions = {}
            for model, backend in backends.items():
                if isinstance(backend, Exception):
                    result = {"error": str(backend), "text": "", "confidence": None, "latency_ms": None}
                else:
                    started = time.perf_counter()
                    try:
                        prediction = backend.recognize(crop)
                        result = {
                            "text": prediction.text.strip(), "confidence": prediction.confidence,
                            "latency_ms": prediction.latency_ms,
                        }
                    except Exception as exc:
                        result = {
                            "error": str(exc), "text": "", "confidence": None,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                element_predictions[model] = result
                per_model[model].append(result)
            element["ocr_predictions"] = element_predictions
            primary = element_predictions.get(models[0], {})
            element["ocr_text"] = primary.get("text", "")
            element["confidence"] = primary.get("confidence") or 0.0

        aggregated = {}
        for model, results in per_model.items():
            confidences = [
                result["confidence"] for result in results
                if isinstance(result.get("confidence"), (int, float))
            ]
            latencies = [
                result["latency_ms"] for result in results
                if isinstance(result.get("latency_ms"), (int, float))
            ]
            aggregated[model] = {
                "text": "".join(result.get("text", "") for result in results),
                "confidence": sum(confidences) / len(confidences) if confidences else None,
                "latency_ms": sum(latencies) if latencies else None,
                "element_count": len(results),
            }
            errors = [result["error"] for result in results if result.get("error")]
            if errors:
                aggregated[model]["errors"] = errors
        return aggregated


    _recognize_digit_elements = _recognize_sequence_elements

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    current, total, label = payload
                    self.status_var.set(f"OCR {current} de {total}: {label}")
                    continue
                if kind == "status":
                    self.status_var.set(str(payload))
                    continue
                self.progress.stop()
                self.run_button.configure(state="normal")
                if kind == "done":
                    manifest_path, output = payload
                    self._load_result(Path(manifest_path), Path(output))
                else:
                    self.status_var.set("O pipeline terminou com erro.")
                    messagebox.showerror("Pipeline", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _load_result(self, manifest_path: Path, output: Path) -> None:
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.output_path = output
            aligned_path = output / "debug" / "aligned.png"
            self.viewer_image = cv2.imread(str(aligned_path), cv2.IMREAD_COLOR)
            self.viewer_dpi = int(self.manifest["page_size"]["dpi"])
            if self.viewer_image is None:
                document_path = Path(self.manifest["source_files"]["completed"])
                self.viewer_image = pdf_to_image(str(document_path), self.manifest["page_size"]["dpi"]).image
            self.viewer_source_image = self.viewer_image.copy()
            document_path = Path(self.document_var.get()).expanduser()
            if document_path.is_file():
                self._update_viewer_info(document_path, self.viewer_source_image)
            if self.use_preset_var.get():
                self._refresh_document_preprocess()
            self._viewer_cache_key = None
            self.hovered_index = None
            self.pinned_index = None
            self._render_viewer()
            total = len(self.manifest.get("fields", []))
            filled = sum(bool(field.get("filled")) for field in self.manifest.get("fields", []))
            models = ", ".join(self.manifest.get("ocr_run", {}).get("models", []))
            self.status_var.set(f"Concluído: {filled} de {total} campos detectados · OCR: {models or 'não executado'}.")
        except Exception as exc:
            self.status_var.set("Resultado gerado, mas não foi possível exibi-lo.")
            messagebox.showerror("Resultado", str(exc))

    def _render_viewer(self) -> None:
        if self.viewer_image is None:
            self.canvas.delete("all")
            self.empty_state.place(relx=0.5, rely=0.5, anchor="center")
            return
        self.empty_state.place_forget()
        h, w = self.viewer_image.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.viewer_scale = min(cw/w, ch/h, 1.0)
        rw, rh = max(1, round(w*self.viewer_scale)), max(1, round(h*self.viewer_scale))
        ox, oy = (cw - rw) // 2, (ch - rh) // 2
        self.viewer_offset = (ox, oy)
        cache_key = (id(self.viewer_image), rw, rh, ox, oy)
        if cache_key != self._viewer_cache_key:
            self.canvas.delete("all")
            rgb = (
                cv2.cvtColor(self.viewer_image, cv2.COLOR_GRAY2RGB)
                if self.viewer_image.ndim == 2
                else cv2.cvtColor(self.viewer_image, cv2.COLOR_BGR2RGB)
            )
            resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA) if (rw, rh) != (w, h) else rgb
            self.viewer_photo = ImageTk.PhotoImage(Image.fromarray(resized))
            self.canvas.create_image(ox, oy, anchor="nw", image=self.viewer_photo)
            self._viewer_cache_key = cache_key
        else:
            self.canvas.delete("overlay")
        if self.manifest:
            for index, field in enumerate(self.manifest.get("fields", [])):
                x1, y1, x2, y2 = (round(value*self.viewer_scale) for value in field["bbox"])
                hovered = index == self.hovered_index
                color = "#ffcc55" if hovered else ("#3ecb8a" if field.get("filled") else "#7c9cff")
                self.canvas.create_rectangle(
                    x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                    outline=color, width=3 if hovered else 2,
                    tags="overlay",
                )
                for element in field.get("elements", []):
                    ex1, ey1, ex2, ey2 = (
                        round(value * self.viewer_scale) for value in element["bbox"]
                    )
                    self.canvas.create_rectangle(
                        ex1 + ox, ey1 + oy, ex2 + ox, ey2 + oy,
                        outline="#3ecb8a", width=2, tags="overlay",
                    )

    def _hover_field(self, event: tk.Event) -> None:
        if self.pinned_index is not None or not self.manifest or self.viewer_scale <= 0:
            return
        hit = self._field_at(event.x, event.y)
        if hit == self.hovered_index:
            return
        self.hovered_index = hit
        self._render_viewer()
        if hit is None:
            self._clear_details()
        else:
            self._show_field(self.manifest["fields"][hit])

    def _field_at(self, canvas_x: float, canvas_y: float) -> int | None:
        if not self.manifest or self.viewer_scale <= 0:
            return None
        ox, oy = self.viewer_offset
        x = (canvas_x - ox) / self.viewer_scale
        y = (canvas_y - oy) / self.viewer_scale
        for index, field in enumerate(self.manifest.get("fields", [])):
            x1, y1, x2, y2 = field["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return index
        return None

    def _pin_field(self, event: tk.Event) -> None:
        hit = self._field_at(event.x, event.y)
        if hit is None:
            self.pinned_index = None
            self.hovered_index = None
            self._render_viewer()
            self._clear_details()
            return
        self.pinned_index = hit
        self.hovered_index = hit
        self._render_viewer()
        self._show_field(self.manifest["fields"][hit])

    def _show_field(self, field: dict) -> None:
        pin = " · fixado" if self.pinned_index is not None else ""
        self.field_title.configure(text=f"{field.get('field_id')} · {field.get('label')}{pin}")
        self.ocr_text.configure(state="normal")
        self.ocr_text.delete("1.0", "end")
        predictions = field.get("ocr_predictions", {})
        if predictions:
            blocks = []
            for model, prediction in predictions.items():
                confidence = prediction.get("confidence")
                confidence_text = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "—"
                latency = prediction.get("latency_ms")
                latency_text = f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "—"
                text = prediction.get("text") or prediction.get("error") or "Nenhum texto reconhecido"
                blocks.append(f"{model} · confiança {confidence_text} · {latency_text}\n{text}")
            displayed_text = "\n\n".join(blocks)
        else:
            displayed_text = field.get("ocr_text") or "OCR não executado para este campo"
        self.ocr_text.insert("1.0", displayed_text)
        self.ocr_text.configure(state="disabled")
        confidence = field.get("confidence")
        confidence_text = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "—"
        elements = field.get("elements", [])
        element_summary = ""
        if elements:
            if field.get("data_type") == "checkbox":
                states = [
                    f"{item.get('label') or item.get('element_id')}"
                    f" ({item.get('value') or item.get('element_id')}): "
                    f"{'marcado' if item.get('filled') else 'vazio'}"
                    for item in elements
                ]
            else:
                states = [
                    f"{item.get('label') or item.get('element_id')}: "
                    f"{item.get('ocr_text') or 'vazio'}"
                    for item in elements
                ]
            element_summary = "\nSubelementos:\n" + "\n".join(states)
        self.field_meta.configure(text=(
            f"Tipo: {field.get('data_type', 'text')}\n"
            f"Preenchido: {'sim' if field.get('filled') else 'não'}\n"
            f"Confiança: {confidence_text}\n"
            f"Detecção: {field.get('detection_method', '—')}\n"
            f"Regex: {(field.get('validation') or {}).get('pattern') or 'nenhuma'}"
            f"{element_summary}"
        ))
        crop_path = field.get("crop_path")
        crop = cv2.imread(str(self.output_path / crop_path), cv2.IMREAD_COLOR) if crop_path and self.output_path else None
        if crop is None:
            self.crop_photo = None
            self.crop_label.configure(image="", text="Recorte indisponível")
        else:
            self.crop_photo = self._crop_photo(crop)
            self.crop_label.configure(image=self.crop_photo, text="")

        self._update_preset_preview()

    @staticmethod
    def _resize_crop(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        scale = min(450/w, 190/h, 1.0)
        size = (max(1, round(w*scale)), max(1, round(h*scale)))
        return cv2.resize(crop, size, interpolation=cv2.INTER_AREA) if size != (w, h) else crop

    def _crop_photo(self, crop: np.ndarray) -> ImageTk.PhotoImage:
        resized = self._resize_crop(crop)
        rgb = (
            cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            if resized.ndim == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        )
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _clear_details(self) -> None:
        self.field_title.configure(text="Passe o mouse ou clique para fixar")
        self.crop_photo = None
        self.crop_label.configure(image="", text="")
        self.processed_crop_photo = None
        self.processed_crop_label.configure(image="", text="")
        self.ocr_text.configure(state="normal")
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.configure(state="disabled")
        self.field_meta.configure(text="")

    def _clear_hover(self) -> None:
        if self.pinned_index is not None:
            return
        if self.hovered_index is not None:
            self.hovered_index = None
            self._render_viewer()
        self._clear_details()
